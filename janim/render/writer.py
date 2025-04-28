import ctypes
import os
import shutil
import subprocess as sp
import time
from contextlib import contextmanager
from functools import partial
from typing import Generator

import OpenGL.GL as gl
from tqdm import tqdm as ProgressDisplay

from janim.anims.timeline import BuiltTimeline, Timeline, TimeRange
from janim.exception import EXITCODE_FFMPEG_NOT_FOUND, ExitException
from janim.locale.i18n import get_local_strings
from janim.logger import log
from janim.render.base import create_context
from janim.render.framebuffer import create_framebuffer, framebuffer_context

_ = get_local_strings('writer')

PBO_COUNT = 3


class VideoWriter:
    '''
    将时间轴动画生成视频输出到文件中

    可以直接调用 ``VideoWriter.writes(MyTimeline().build())`` 进行输出

    主要流程在 :meth:`write_all` 中：

    - 首先调用 ffmpeg，这里用它生成视频（先输出到 _temp 文件中）
    - 然后遍历动画的每一帧，进行渲染，并将像素数据传递给 ffmpeg
    - 最后结束 ffmpeg 的调用，完成 _temp 文件的输出
    - 将 _temp 文件改名，删去 "_temp" 后缀，完成视频输出
    '''
    def __init__(self, built: BuiltTimeline):
        self.built = built
        try:
            self.ctx = create_context(standalone=True, require=430)
        except ValueError:
            self.ctx = create_context(standalone=True, require=330)

        pw, ph = built.cfg.pixel_width, built.cfg.pixel_height
        self.frame_count = round(built.duration * built.cfg.fps) + 1
        self.fbo = create_framebuffer(self.ctx, pw, ph)
        # 创建用于后处理的FBO
        self.post_fbo = create_framebuffer(self.ctx, pw, ph)
        # 创建后处理shader
        self._init_post_process_shader()

        # PBO 相关初始化
        self.byte_size = pw * ph * 4  # 每帧的字节大小 (RGBA)

    def _init_nvenc(self, file_path: str) -> bool:
        """
        初始化NVIDIA编码器

        返回:
            bool: 初始化是否成功
        """
        # 查找DLL路径 - 首先检查当前目录，然后检查库目录
        dll_paths = [
            "GlCudaNvEncoder.dll",  # 当前目录
            os.path.join(os.path.dirname(__file__), "GlCudaNvEncoder.dll"),  # 渲染模块目录
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib", "GlCudaNvEncoder.dll"),  # 库目录
        ]

        dll_path = None
        for path in dll_paths:
            if os.path.exists(path):
                dll_path = path
                break

        if not dll_path:
            log.warning(_("NVIDIA encoder DLL not found, falling back to FFmpeg"))
            return False

        try:
            # 添加DLL所在目录到PATH环境变量，确保所有依赖项都能被找到
            os.environ['PATH'] = os.path.dirname(os.path.abspath(dll_path)) + os.pathsep + os.environ['PATH']

            try:
                # 尝试使用windll加载器
                self.nvenc_dll = ctypes.windll.LoadLibrary(dll_path)
                log.info(_("Successfully loaded NVIDIA encoder DLL using windll"))
            except Exception as e1:
                # log.warning(_("Using windll加载DLL失败: {0}，尝试使用CDLL").format(str(e1)))
                try:
                    # 如果windll失败，尝试使用CDLL
                    self.nvenc_dll = ctypes.CDLL(os.path.abspath(dll_path))
                    log.info(_("Successfully loaded NVIDIA encoder DLL using CDLL"))
                except Exception as e2:
                    log.error(_("All DLL loading methods failed: {0}").format(str(e2)))
                    return False

            # 定义函数原型
            self.nvenc_dll.gcne_create_encoder.restype = ctypes.c_void_p
            self.nvenc_dll.gcne_create_encoder.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

            self.nvenc_dll.gcne_register_texture.restype = ctypes.c_int
            self.nvenc_dll.gcne_register_texture.argtypes = [ctypes.c_void_p, ctypes.c_uint]

            self.nvenc_dll.gcne_encode_frame.restype = ctypes.c_int
            self.nvenc_dll.gcne_encode_frame.argtypes = [
                ctypes.c_void_p,                        # state
                ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),  # packet_data
                ctypes.POINTER(ctypes.c_int)            # packet_size
            ]

            self.nvenc_dll.gcne_free_packet.restype = None
            self.nvenc_dll.gcne_free_packet.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]

            self.nvenc_dll.gcne_destroy_encoder.restype = ctypes.c_int
            self.nvenc_dll.gcne_destroy_encoder.argtypes = [
                ctypes.c_void_p,                        # state
                ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),  # packet_data
                ctypes.POINTER(ctypes.c_int)            # packet_size
            ]

            # 创建编码器实例
            pw, ph = self.built.cfg.pixel_width, self.built.cfg.pixel_height

            # 使用GPU ID 0（通常是默认GPU）
            self.nvenc_encoder = self.nvenc_dll.gcne_create_encoder(pw, ph, 0)

            if not self.nvenc_encoder:
                log.warning(_("Failed to create NVIDIA encoder, falling back to FFmpeg"))
                return False

            # 注册OpenGL纹理
            texture_id = self.post_fbo.color_attachments[0].glo
            result = self.nvenc_dll.gcne_register_texture(self.nvenc_encoder, texture_id)
            if result != 0:
                error_msg = _("Failed to register texture with NVIDIA encoder (error code: {0}), falling back to FFmpeg").format(result)
                log.warning(error_msg)

                # 使用NULL指针销毁编码器
                self.nvenc_dll.gcne_destroy_encoder(
                    self.nvenc_encoder,
                    ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))(),
                    ctypes.POINTER(ctypes.c_int)()
                )
                self.nvenc_encoder = None
                return False

            ffmpeg_cmd = [
                self.built.cfg.ffmpeg_bin,
                '-y',                # 覆盖输出文件
                '-f', 'h264',        # 输入格式是H.264
                '-i', '-',           # 从标准输入读取
                '-c:v', 'copy',      # 直接复制视频流，不重新编码
                '-r', str(self.built.cfg.fps),  # 设置帧率
                file_path,           # 输出文件
                '-loglevel', 'error'  # 只显示错误信息
            ]

            cmd_str = ' '.join(ffmpeg_cmd)
            try:
                self.nvenc_ffmpeg_pipe = sp.Popen(
                    ffmpeg_cmd,
                    stdin=sp.PIPE,
                    stderr=sp.PIPE
                )
                self.nvenc_file_path = file_path
                return True
            except Exception as e:
                log.error(_("Failed to create FFmpeg pipe: {0}").format(str(e)))

                # 销毁编码器
                self.nvenc_dll.gcne_destroy_encoder(
                    self.nvenc_encoder,
                    ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))(),
                    ctypes.POINTER(ctypes.c_int)()
                )
                self.nvenc_encoder = None
                return False

        except Exception as e:
            log.warning(_("Error initializing NVIDIA encoder: {0}, falling back to FFmpeg").format(str(e)))
            return False

    def _cleanup_nvenc(self):
        """清理NVIDIA编码器资源，并处理最终输出"""
        if hasattr(self, 'nvenc_encoder') and self.nvenc_encoder:
            # 销毁编码器，并获取最后的数据包

            # 准备接收最终数据包的变量
            packet_data_ptr = ctypes.POINTER(ctypes.c_ubyte)()
            packet_size = ctypes.c_int(0)

            # 调用销毁编码器函数，获取最后的数据包
            result = self.nvenc_dll.gcne_destroy_encoder(
                self.nvenc_encoder,
                ctypes.byref(packet_data_ptr),
                ctypes.byref(packet_size)
            )

            if result == 0 and packet_data_ptr and packet_size.value > 0:
                # 如果有最终数据包，将其写入FFmpeg管道
                final_data = ctypes.string_at(packet_data_ptr, packet_size.value)
                self.nvenc_ffmpeg_pipe.stdin.write(final_data)

                # 释放数据包内存
                self.nvenc_dll.gcne_free_packet(packet_data_ptr)

            # 关闭FFmpeg管道
            self.nvenc_ffmpeg_pipe.stdin.close()
            self.nvenc_ffmpeg_pipe.wait()
            self.nvenc_encoder = None

    def _init_post_process_shader(self):
        """
        由于OpenGL的坐标系和纹理坐标系不同，OpenGL的坐标系是左下角为原点，而纹理坐标系是左上角为原点
        因此在渲染时需要对纹理坐标进行垂直翻转（vflip），否则渲染出来的图像会是上下颠倒的
        这里使用shader提前施加垂直翻转以解决问题
        """
        vertex_shader = """
        #version 330
        in vec2 in_vert;
        in vec2 in_texcoord;
        out vec2 v_texcoord;

        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            v_texcoord = vec2(in_texcoord.x, 1.0 - in_texcoord.y); // vflip
        }
        """

        fragment_shader = """
        #version 330
        uniform sampler2D texture0;
        in vec2 v_texcoord;
        out vec4 f_color;

        void main() {
            vec4 color = texture(texture0, v_texcoord);
            f_color = color;
        }
        """

        # 创建着色器程序
        self.post_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        # 创建一个简单的全屏幕四边形用于渲染
        vertices = [
            # x, y, u, v
            -1.0, -1.0, 0.0, 0.0,  # 左下
            -1.0,  1.0, 0.0, 1.0,  # 左上
             1.0, -1.0, 1.0, 0.0,  # 右下
             1.0,  1.0, 1.0, 1.0,  # 右上
        ]
        # numpy is not required, use array module to create float array
        import array
        vertices_array = array.array('f', vertices)
        self.post_quad = self.ctx.buffer(vertices_array)
        self.post_vao = self.ctx.vertex_array(
            self.post_program,
            [
                (self.post_quad, '2f 2f', 'in_vert', 'in_texcoord'),
            ],
        )

    def _apply_post_process(self):
        """应用后处理效果（垂直翻转和颜色转换）"""
        # 绑定后处理FBO
        with framebuffer_context(self.post_fbo):
            # 清除后处理FBO
            self.post_fbo.clear(0.0, 0.0, 0.0, 0.0)

            # 设置源纹理（原始FBO的颜色附件）
            self.fbo.color_attachments[0].use(location=0)

            # 绘制全屏四边形，应用shader
            self.post_vao.render(gl.GL_TRIANGLE_STRIP)

    def _init_pbos(self) -> None:
        '''初始化PBO缓冲区'''
        self.pbos = gl.glGenBuffers(PBO_COUNT)

        for pbo in self.pbos:
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, pbo)
            # 分配空间，GL_STREAM_READ 表明数据将从 GPU 读取到 CPU，并且每帧都会更新
            gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, self.byte_size, None, gl.GL_STREAM_READ)

        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)  # 解绑PBO

    def _read_idx_iter(self) -> Generator[int | None, None, None]:
        for _ in range(PBO_COUNT - 1):
            yield None
        for frame_idx in range(self.frame_count):
            yield frame_idx % PBO_COUNT

    def _cleanup_pbos(self) -> None:
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)  # 确保解绑
        # 正确删除多个缓冲区
        gl.glDeleteBuffers(len(self.pbos), self.pbos)

    @staticmethod
    def writes(built: BuiltTimeline, file_path: str, *, quiet=False, use_pbo=True, hwaccel=False, on_device=False) -> None:
        VideoWriter(built).write_all(file_path, quiet=quiet, use_pbo=use_pbo, hwaccel=hwaccel, on_device=on_device)

    def write_all(self, file_path: str, *, quiet=False, use_pbo=True, hwaccel=False, on_device=False, _keep_temp=False) -> None:
        '''将时间轴动画输出到文件中

        - 指定 ``quiet=True``，则不会输出前后的提示信息，但仍有进度条
        '''
        name = self.built.timeline.__class__.__name__
        if not quiet:
            log.info(_('Writing video "{name}"').format(name=name))
            t = time.time()

        fps = self.built.cfg.fps
        rgb = self.built.cfg.background_color.rgb

        if on_device and self._init_nvenc(file_path):
            # 使用 NVIDIA 硬件编码器
            log.info(_("Using NVIDIA hardware encoding"))
            progress_display = ProgressDisplay(
                range(self.frame_count),
                leave=False,
                dynamic_ncols=True
            )
            with framebuffer_context(self.fbo):
                for frame in progress_display:
                    # 渲染当前帧到 FBO
                    self.fbo.clear(*rgb, True)
                    self.built.render_all(self.ctx, frame / fps, blend_on=True)

                    # 应用后处理（垂直翻转和颜色通道调整）
                    self._apply_post_process()

                    # 创建指针来接收编码数据包
                    packet_data_ptr = ctypes.POINTER(ctypes.c_ubyte)()
                    packet_size = ctypes.c_int(0)

                    # 编码当前帧
                    result = self.nvenc_dll.gcne_encode_frame(
                        self.nvenc_encoder,
                        ctypes.byref(packet_data_ptr),
                        ctypes.byref(packet_size)
                    )

                    if result != 0:
                        log.error(_("Error encoding frame {0} using NVIDIA encoder").format(frame))
                    elif packet_data_ptr and packet_size.value > 0:
                        # 如果有数据包，将其写入到FFmpeg管道
                        frame_data = ctypes.string_at(packet_data_ptr, packet_size.value)
                        self.nvenc_ffmpeg_pipe.stdin.write(frame_data)

                        # 释放数据包内存
                        self.nvenc_dll.gcne_free_packet(packet_data_ptr)

                    # 每隔一段帧数刷新FFmpeg管道，避免缓冲区溢出
                    if frame % 30 == 0:
                        self.nvenc_ffmpeg_pipe.stdin.flush()

            # 清理编码器资源并处理最终输出
            self._cleanup_nvenc()
        else:
            # 使用常规的 FFmpeg 编码
            if on_device:
                log.info(_("NVIDIA encoder not available, falling back to FFmpeg"))

            self.open_video_pipe(file_path, hwaccel)
            progress_display = ProgressDisplay(
                range(self.frame_count),
                leave=False,
                dynamic_ncols=True
            )
            transparent = self.ext == '.mov'
            if use_pbo:
                self._init_pbos()

                # 使用PBO优化的渲染循环
                with framebuffer_context(self.fbo):
                    read_idx_iter = self._read_idx_iter()
                    for frame_idx, read_idx in zip(progress_display, read_idx_iter):
                        # 渲染当前帧
                        self.fbo.clear(*rgb, not transparent)
                        if transparent:
                            gl.glFlush()
                        self.built.render_all(self.ctx, frame_idx / fps, blend_on=not transparent)

                        # 绑定当前PBO来存储新帧
                        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.pbos[frame_idx % PBO_COUNT])
                        # 注意: 当PBO绑定时，最后一个参数是偏移量而不是指针
                        gl.glReadPixels(0, 0, self.built.cfg.pixel_width, self.built.cfg.pixel_height,
                                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, 0)

                        # 如果不是第一批，处理上一批的数据
                        if read_idx is not None:
                            # 绑定对应的PBO，用于读取数据
                            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.pbos[read_idx])

                            # 使用numpy从映射的内存中读取数据
                            ptr = gl.glMapBuffer(gl.GL_PIXEL_PACK_BUFFER, gl.GL_READ_ONLY)
                            assert ptr
                            data = gl.ctypes.string_at(ptr, self.byte_size)
                            # 写入数据到ffmpeg
                            self.writing_process.stdin.write(data)
                            gl.glUnmapBuffer(gl.GL_PIXEL_PACK_BUFFER)

                    # 处理最后一批
                    for read_idx in read_idx_iter:
                        # 在大多数情况下 read_idx 并不是 None
                        # 只有在 Timeline 时长特别短的时候会出现 None
                        if read_idx is None:
                            continue
                        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self.pbos[read_idx])
                        data = gl.glGetBufferSubData(gl.GL_PIXEL_PACK_BUFFER, 0, self.byte_size)
                        self.writing_process.stdin.write(data)

                self._cleanup_pbos()
            else:
                # 原始渲染循环（不使用PBO）
                with framebuffer_context(self.fbo):
                    for frame in progress_display:
                        self.fbo.clear(*rgb, not transparent)
                        # 在输出 mov 时，framebuffer 是透明的
                        # 为了颜色能被正确渲染到透明 framebuffer 上
                        # 这里需要禁用自带 blending 的并使用 shader 里自定义的 blending（参考 program.py 的 injection_ja_finish_up）
                        # 但是 shader 里的 blending 依赖 framebuffer 信息
                        # 所以这里需要使用 glFlush 更新 framebuffer 信息使得正确渲染
                        if transparent:
                            gl.glFlush()
                        self.built.render_all(self.ctx, frame / fps, blend_on=not transparent)
                        bytes = self.fbo.read(components=4)
                        self.writing_process.stdin.write(bytes)

            self.close_video_pipe(_keep_temp)

        if not quiet:
            log.info(
                _('Finished writing video "{name}" in {elapsed:.2f} s')
                .format(name=name, elapsed=time.time() - t)
            )

            if not _keep_temp:
                log.info(
                    _('File saved to "{file_path}" (video only)')
                    .format(file_path=file_path)
                )

    def open_video_pipe(self, file_path: str, hwaccel: bool) -> None:
        stem, self.ext = os.path.splitext(file_path)
        self.final_file_path = file_path
        self.temp_file_path = stem + '_temp' + self.ext

        command = [
            self.built.cfg.ffmpeg_bin,
            '-y',   # overwrite output file if it exists
            '-f', 'rawvideo',
            '-s', f'{self.built.cfg.pixel_width}x{self.built.cfg.pixel_height}',  # size of one frame
            '-pix_fmt', 'rgba',
            '-r', str(self.built.cfg.fps),  # frames per second
            '-i', '-',  # The input comes from a pipe
            '-vf', 'vflip',
            '-an',  # Tells FFMPEG not to expect any audio
            '-loglevel', 'error',
        ]

        if self.ext == '.mp4':
            command += [
                '-pix_fmt', 'yuv420p',
                '-vcodec', self.find_encoder(self.built.cfg.ffmpeg_bin, hwaccel),
            ]
        elif self.ext == '.mov':
            # This is if the background of the exported
            # video should be transparent.
            command += [
                '-vcodec', 'qtrle',
            ]
        elif self.ext == '.gif':
            pass
        else:
            assert False

        command += [self.temp_file_path]
        with self.handle_ffmpeg_not_found():
            self.writing_process = sp.Popen(command, stdin=sp.PIPE)

    hwencoder_cache: str | None = None

    @staticmethod
    def find_encoder(ffmpeg_bin: str, hwaccel: bool) -> str:
        '''查找编码器，若 ``hwaccel=True`` 则优先使用硬件编码器'''
        if not hwaccel:
            encoder = 'libx264'
        else:
            if VideoWriter.hwencoder_cache is not None:
                encoder = VideoWriter.hwencoder_cache
            else:
                # call ffmpeg to test nvenc/amf support
                with VideoWriter.handle_ffmpeg_not_found():
                    test_availability = sp.Popen(
                        [ffmpeg_bin, '-hide_banner', '-encoders'],
                        stdout=sp.PIPE,
                        stderr=sp.PIPE
                    )

                out, err = test_availability.communicate()
                if b'h264_nvenc' in out:
                    encoder = 'h264_nvenc'
                elif b'h264_amf' in out:
                    encoder = 'h264_amf'
                else:
                    encoder = 'libx264'
                    log.info(_('No hardware encoder found'))
                VideoWriter.hwencoder_cache = encoder

        log.info(_('Using {encoder} for encoding').format(encoder=encoder))
        return encoder

    def close_video_pipe(self, _keep_temp: bool) -> None:
        self.writing_process.stdin.close()
        self.writing_process.wait()
        self.writing_process.terminate()
        if not _keep_temp:
            shutil.move(self.temp_file_path, self.final_file_path)

    @staticmethod
    @contextmanager
    def handle_ffmpeg_not_found():
        try:
            yield
        except FileNotFoundError:
            log.error(_('Unable to output video. '
                        'Please install ffmpeg and add it to the environment variables.'))
            raise ExitException(EXITCODE_FFMPEG_NOT_FOUND)


class AudioWriter:
    def __init__(self, built: BuiltTimeline):
        self.built = built

    @staticmethod
    def writes(built: BuiltTimeline, file_path: str, *, quiet=False) -> None:
        AudioWriter(built).write_all(file_path, quiet=quiet)

    def write_all(self, file_path: str, *, quiet=False, _keep_temp=False) -> None:
        name = self.built.timeline.__class__.__name__
        if not quiet:
            log.info(_('Writing audio of "{name}"').format(name=name))
            t = time.time()

        fps = self.built.cfg.fps
        framerate = self.built.cfg.audio_framerate

        self.open_audio_pipe(file_path)

        progress_display = ProgressDisplay(
            range(round(self.built.duration * fps) + 1),
            leave=False,
            dynamic_ncols=True
        )

        get_audio_samples = partial(self.built.get_audio_samples_of_frame,
                                    fps,
                                    framerate)

        for frame in progress_display:
            samples = get_audio_samples(frame)
            self.writing_process.stdin.write(samples.tobytes())

        self.close_audio_pipe(_keep_temp)

        if not quiet:
            log.info(
                _('Finished writing audio of "{name}" in {elapsed:.2f} s')
                .format(name=name, elapsed=time.time() - t)
            )

            if not _keep_temp:
                log.info(
                    _('File saved to "{file_path}"')
                    .format(file_path=file_path)
                )

    def open_audio_pipe(self, file_path: str) -> None:
        stem, ext = os.path.splitext(file_path)
        self.final_file_path = file_path
        self.temp_file_path = stem + '_temp' + ext

        command = [
            self.built.cfg.ffmpeg_bin,
            '-y',   # overwrite output file if it exists
            '-f', 's16le',
            '-ar', str(self.built.cfg.audio_framerate),      # framerate & samplerate
            '-ac', str(self.built.cfg.audio_channels),
            '-i', '-',
            '-loglevel', 'error',
            self.temp_file_path
        ]

        try:
            self.writing_process = sp.Popen(command, stdin=sp.PIPE)
        except FileNotFoundError:
            log.error(_('Unable to output audio. '
                        'Please install ffmpeg and add it to the environment variables.'))
            raise ExitException(EXITCODE_FFMPEG_NOT_FOUND)

    def close_audio_pipe(self, _keep_temp: bool) -> None:
        self.writing_process.stdin.close()
        self.writing_process.wait()
        self.writing_process.terminate()
        if not _keep_temp:
            shutil.move(self.temp_file_path, self.final_file_path)


def merge_video_and_audio(
    ffmpeg_bin: str,
    video_path: str,
    audio_path: str,
    result_path: str,
    remove: bool = True,
    *,
    quiet: bool = False,
) -> None:
    command = [
        ffmpeg_bin,
        '-y',
        '-i', video_path,
        '-i', audio_path,
        '-shortest',
        '-c:v', 'copy',
        '-c:a', 'aac',
        result_path,
        '-loglevel', 'error'
    ]

    try:
        merge_process = sp.Popen(command, stdin=sp.PIPE)
    except FileNotFoundError:
        log.error(_('Unable to merge video. '
                    'Please install ffmpeg and add it to the environment variables.'))
        raise ExitException(EXITCODE_FFMPEG_NOT_FOUND)

    merge_process.wait()
    merge_process.terminate()

    if remove:
        os.remove(video_path)
        os.remove(audio_path)

    if not quiet:
        log.info(
            _('File saved to "{file_path}" (merged)')
            .format(file_path=result_path)
        )


class SRTWriter:
    @staticmethod
    def writes(built: BuiltTimeline, file_path: str) -> None:
        with open(file_path, 'wt') as file:
            chunks: list[tuple[TimeRange, list[Timeline.SubtitleInfo]]] = []

            for info in built.timeline.subtitle_infos:
                if not chunks or chunks[-1][0] != info.range:
                    chunks.append((info.range, []))
                chunks[-1][1].append(info)

            for i, chunk in enumerate(chunks, start=1):
                file.write(f'\n{i}\n')
                file.write(f'{SRTWriter.t_to_srt_time(chunk[0].at)} --> {SRTWriter.t_to_srt_time(chunk[0].end)}\n')
                for info in reversed(chunk[1]):
                    file.write(f'{info.text}\n')

    @staticmethod
    def t_to_srt_time(t: float):
        '''
        将秒数转换为 SRT 时间格式：HH:MM:SS,mmm
        '''
        t = round(t, 3)
        hours = int(t // 3600)
        minutes = int((t % 3600) // 60)
        secs = int(t % 60)
        millis = int((t % 1) * 1000)

        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"