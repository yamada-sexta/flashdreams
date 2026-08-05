# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Non-streaming Wan 2.1 runner classes (T2V and I2V)."""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated

import torch
import tyro
from loguru import logger
from tyro.constructors import PrimitiveConstructorSpec

from flashdreams.core.attention import Int4BlockKVCache
from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.infra.runner_io import (
    ensure_output_dir,
    load_first_frame_tensor,
    read_image_rgb,
    resolve_input_path,
    resolve_prompt_value,
    runner_artifact_path,
    write_runner_stats,
    write_video_tensor,
)
from flashdreams.recipes.wan import (
    NEGATIVE_PROMPT,
    WanInferencePipeline,
    WanInferencePipelineCache,
    WanInferencePipelineConfig,
)

__all__ = [
    "Wan21I2VRunnerConfig",
    "Wan21I2VRunner",
    "Wan21T2VRunnerConfig",
    "Wan21T2VRunner",
]


DEFAULT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on "
    "a surfboard. The fluffy-furred feline gazes directly at the camera "
    "with a relaxed expression. Blurred beach scenery forms the background "
    "featuring crystal-clear waters, distant green hills, and a blue sky "
    "dotted with white clouds. The cat assumes a naturally relaxed posture, "
    "as if savoring the sea breeze and warm sunlight. A close-up shot "
    "highlights the feline's intricate details and the refreshing "
    "atmosphere of the seaside."
)

DEFAULT_I2V_IMAGE_URL = (
    "https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/examples/i2v_input.JPG"
)

IMAGE_CACHE_DIR = (
    Path(os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")))
    / "wan21"
)
"""User-writable cache for on-the-fly I2V first-frame downloads."""


_LOW_VRAM_FLAG = PrimitiveConstructorSpec(
    nargs=0,
    metavar="",
    instance_from_str=lambda _: True,
    is_instance=lambda value: isinstance(value, bool),
    str_from_instance=lambda _: [],
)

_LOW_VRAM_DECODE_CHUNK_SIZE = 1
"""Latent frames decoded per VAE call in low-VRAM mode."""


@dataclass(kw_only=True)
class Wan21T2VRunnerConfig(RunnerConfig):
    """Runner config for the Wan 2.1 T2V variant.

    Also serves as the base for :class:`Wan21I2VRunnerConfig`
    (I2V is T2V plus an ``image_path``).
    """

    _target: type["Wan21T2VRunner"] = field(default_factory=lambda: Wan21T2VRunner)

    prompt: str | Path = DEFAULT_PROMPT
    """Either an inline text prompt (--prompt "...") or a path to a
    txt file whose first line is read as the prompt (--prompt prompt.txt).
    Defaults to :data:`DEFAULT_PROMPT`."""

    pixel_height: int = 480
    """Output video pixel height."""

    pixel_width: int = 832
    """Output video pixel width."""

    fps: int = 16
    """Output video frame rate."""

    postprocess_output_layout: VideoTensorLayout | None = "tchw"
    """Pipeline output layout for streaming post-processing."""

    low_vram: Annotated[bool, _LOW_VRAM_FLAG] = False
    """Stage the native UMT5, DiT, and VAE for a 6 GiB GPU."""

    low_vram_pipeline: Annotated[
        WanInferencePipelineConfig | None, tyro.conf.Suppress
    ] = None
    """Internal staged variant selected when :attr:`low_vram` is enabled."""

    gpu_memory_budget_gib: float = 6.0
    """PyTorch allocator ceiling used by low-VRAM mode."""


@dataclass(kw_only=True)
class Wan21I2VRunnerConfig(Wan21T2VRunnerConfig):
    """Runner config for the Wan 2.1 I2V variant.

    Inherits all T2V fields (prompt, pixel_*, fps) and
    adds the first-frame image path that I2V needs at runtime.
    """

    _target: type["Wan21I2VRunner"] = field(default_factory=lambda: Wan21I2VRunner)

    image_path: str | Path = DEFAULT_I2V_IMAGE_URL
    """Path to the first-frame RGB image, or an ``http(s)://`` URL that
    will be downloaded on first use into :data:`IMAGE_CACHE_DIR`.
    Defaults to :data:`DEFAULT_I2V_IMAGE_URL`."""

    prompt: str | Path = DEFAULT_PROMPT
    """Either an inline text prompt (--prompt "...") or a path to a
    txt file whose first line is read as the prompt (--prompt prompt.txt).
    Defaults to :data:`DEFAULT_PROMPT`."""

    pixel_height: int = 832
    """Output video pixel height."""

    pixel_width: int = 480
    """Output video pixel width."""


class Wan21T2VRunner(Runner[Wan21T2VRunnerConfig, WanInferencePipeline]):
    """Wan 2.1 non-streaming T2V driver.

    Also serves as the base for :class:`Wan21I2VRunner` (I2V
    only overrides :meth:`_initialize_cache` to load the first frame;
    everything else, including :meth:`run`, is reused).
    """

    config: Wan21T2VRunnerConfig

    def __init__(self, config: Wan21T2VRunnerConfig) -> None:
        if config.low_vram:
            assert config.low_vram_pipeline is not None, (
                "low_vram=True requires an internal low_vram_pipeline config"
            )
            config = replace(
                config,
                pipeline=config.low_vram_pipeline,
            )
        super().__init__(config, move_pipeline_to_device=not config.low_vram)

    def _resolve_prompt(self) -> str:
        """Resolve config.prompt.

        A Path reads its first non-empty line, a str is used as-is.
        """
        return resolve_prompt_value(self.config.prompt)

    def _initialize_cache(self) -> WanInferencePipelineCache:
        """Initialize the autoregressive cache for T2V."""
        config = self.config
        prompt = self._resolve_prompt()

        assert isinstance(self.pipeline.decoder, StreamingVideoDecoder)
        sp = self.pipeline.decoder.spatial_compression_ratio
        assert config.pixel_height % sp == 0, (
            f"pixel_height={config.pixel_height} must divide {sp}."
        )
        assert config.pixel_width % sp == 0, (
            f"pixel_width={config.pixel_width} must divide {sp}."
        )
        latent_h = config.pixel_height // sp
        latent_w = config.pixel_width // sp

        return self.pipeline.initialize_cache(
            text=[prompt], image=None, height=latent_h, width=latent_w
        )

    def run(self) -> None:
        """Drive the single-step rollout and write outputs."""
        if self.config.low_vram:
            _run_low_vram(self)
            return
        config = self.config

        # Initialize the autoregressive cache.
        cache = self._initialize_cache()

        # Generate the output in one AR step.
        postprocess_stream = self.create_postprocess_stream(fps=config.fps)
        generated = self.pipeline.generate(autoregressive_index=0, cache=cache)
        stats = self.pipeline.finalize(autoregressive_index=0, cache=cache)
        postprocess_stream.process(generated, autoregressive_index=0)
        generated = postprocess_stream.finish()
        if generated is None:
            return

        # Write the video.
        ensure_output_dir(config.output_dir)
        video_path = runner_artifact_path(config.output_dir, config.runner_name, "mp4")
        write_video_tensor(generated, video_path, fps=config.fps, layout="tchw")

        logger.info(
            f"[{config.runner_name}] wrote video {tuple(generated.shape)} "
            f"-> {video_path.resolve()}"
        )

        # Write the perf stats.
        if stats is not None:
            stats_path = write_runner_stats(
                config.output_dir,
                config.runner_name,
                [{"autoregressive_index": 0, **stats}],
            )
            logger.info(
                f"[{config.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )


class Wan21I2VRunner(Wan21T2VRunner):
    """Wan 2.1 non-streaming I2V driver (first-frame injection)."""

    config: Wan21I2VRunnerConfig

    def _initialize_cache(self) -> WanInferencePipelineCache:
        """Initialize the autoregressive cache for I2V (loads first frame)."""
        config = self.config
        prompt = self._resolve_prompt()

        assert isinstance(self.pipeline.decoder, StreamingVideoDecoder)
        sp = self.pipeline.decoder.spatial_compression_ratio
        assert config.pixel_height % sp == 0, (
            f"pixel_height={config.pixel_height} must divide {sp}."
        )
        assert config.pixel_width % sp == 0, (
            f"pixel_width={config.pixel_width} must divide {sp}."
        )

        # Load + resize the first frame, then convert to [-1, 1] bf16
        # in shape [T=1, C, H, W] (matches batch_shape=()). Pin to the
        # pipeline's actual device so non-default ``--device`` selections
        # (and the auto cuda:LOCAL_RANK override under torchrun) both work.
        image = load_first_frame_tensor(
            resolve_input_path(
                config.image_path,
                cache_dir=IMAGE_CACHE_DIR,
                validator=read_image_rgb,
            ),
            pixel_height=config.pixel_height,
            pixel_width=config.pixel_width,
            device=self.pipeline.device,
            dtype=torch.bfloat16,
        )

        return self.pipeline.initialize_cache(text=[prompt], image=image)


def _cuda_device_index(runner: Wan21T2VRunner) -> int:
    """Resolve a concrete index for CUDA memory APIs."""
    assert runner.device.type == "cuda", "Low-VRAM mode requires CUDA."
    return (
        runner.device.index
        if runner.device.index is not None
        else torch.cuda.current_device()
    )


def _release_cuda_stage(module: torch.nn.Module) -> None:
    """Move a completed stage to CPU and release cached CUDA blocks."""
    module.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


def _int4_cache_stats(cache: WanInferencePipelineCache) -> tuple[int, int, int, int]:
    """Return packed bytes, BF16 bytes, committed tokens, and cache count."""
    transformer_cache = cache.transformer_cache
    network_caches = [transformer_cache.network_cache]
    if transformer_cache.network_cache_uncond is not None:
        network_caches.append(transformer_cache.network_cache_uncond)
    packed_bytes = 0
    uncompressed_bytes = 0
    committed_tokens: list[int] = []
    for network_cache in network_caches:
        for block_cache in network_cache.block_caches:
            if isinstance(block_cache.self_attn, Int4BlockKVCache):
                packed_bytes += block_cache.self_attn.storage_nbytes
                uncompressed_bytes += block_cache.self_attn.uncompressed_nbytes
                committed_tokens.append(block_cache.self_attn._n_cached)
    tokens_per_block = min(committed_tokens, default=0)
    assert all(tokens == tokens_per_block for tokens in committed_tokens), (
        "INT4 block caches must commit the same token count"
    )
    return packed_bytes, uncompressed_bytes, tokens_per_block, len(committed_tokens)


@torch.no_grad()
def _decode_low_vram_chunks(
    decoder: StreamingVideoDecoder,
    clean_latent: torch.Tensor,
    decoder_cache: object,
    device: torch.device,
    latent_chunk_size: int = _LOW_VRAM_DECODE_CHUNK_SIZE,
) -> torch.Tensor:
    """Decode bounded latent chunks while preserving VAE temporal state."""
    assert latent_chunk_size > 0, "latent_chunk_size must be positive"
    chunks: list[torch.Tensor] = []
    num_latent_frames = clean_latent.shape[-4]
    for chunk_index, latent_start in enumerate(
        range(0, num_latent_frames, latent_chunk_size)
    ):
        chunk_length = min(latent_chunk_size, num_latent_frames - latent_start)
        latent_chunk = clean_latent.narrow(-4, latent_start, chunk_length).to(
            device=device,
            dtype=torch.bfloat16,
        )
        decoded_chunk = decoder(
            input=latent_chunk,
            autoregressive_index=chunk_index,
            cache=decoder_cache,
        )
        chunks.append(decoded_chunk.to("cpu"))
        del latent_chunk, decoded_chunk
        decoded_latent_frames = latent_start + chunk_length
        if decoded_latent_frames % 4 == 0 or decoded_latent_frames == num_latent_frames:
            logger.info(
                f"VAE decoded {decoded_latent_frames}/{num_latent_frames} latent frames"
            )
    return torch.cat(chunks, dim=-4)


@torch.no_grad()
def _run_low_vram(runner: Wan21T2VRunner) -> None:
    """Run the native FlashDreams pipeline with staged module residency."""
    config = runner.config
    device_index = _cuda_device_index(runner)
    total_gib = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
    budget = config.gpu_memory_budget_gib
    assert 0.0 < budget <= total_gib, (
        f"gpu_memory_budget_gib must be in (0, {total_gib:.2f}], got {budget}"
    )
    torch.cuda.set_per_process_memory_fraction(budget / total_gib, device_index)

    text_start = time.perf_counter()
    prompt = resolve_prompt_value(config.prompt)
    transformer_config = runner.pipeline.diffusion_model.transformer.config
    guidance_scale = getattr(transformer_config, "guidance_scale", 1.0)
    text_encoder = runner.pipeline.text_encoder
    assert text_encoder is not None, "Low-VRAM mode requires the native UMT5 encoder."
    try:
        text_embeddings = text_encoder([prompt]).to("cpu")
        negative_text_embeddings = (
            text_encoder([NEGATIVE_PROMPT]).to("cpu") if guidance_scale > 1.0 else None
        )
    finally:
        runner.pipeline.release_oneshot_encoders()
    text_seconds = time.perf_counter() - text_start
    logger.info(
        f"[{config.runner_name}] FlashDreams UMT5 encoded "
        f"{tuple(text_embeddings.shape)} "
        f"in {text_seconds:.2f} s"
    )

    assert isinstance(runner.pipeline.decoder, StreamingVideoDecoder)
    decoder = runner.pipeline.decoder
    spatial_ratio = decoder.spatial_compression_ratio
    latent_h = config.pixel_height // spatial_ratio
    latent_w = config.pixel_width // spatial_ratio

    diffusion = runner.pipeline.diffusion_model
    torch.cuda.reset_peak_memory_stats(device_index)
    diffusion.to(runner.device)
    diffuse_start = time.perf_counter()
    cache = runner.pipeline.initialize_cache_from_embeddings(
        text_embeddings,
        height=latent_h,
        width=latent_w,
        negative_text_embeddings=negative_text_embeddings,
    )
    clean_latent, final_state = diffusion.generate(
        autoregressive_index=0,
        cache=cache.transformer_cache,
    )
    diffusion.finalize(final_state)
    (
        kv_cache_bytes,
        kv_cache_uncompressed_bytes,
        kv_cache_tokens_per_block,
        kv_cache_block_count,
    ) = _int4_cache_stats(cache)
    torch.cuda.synchronize(device_index)
    diffuse_seconds = time.perf_counter() - diffuse_start
    diffuse_peak_gib = torch.cuda.max_memory_allocated(device_index) / 1024**3
    diffuse_peak_reserved_gib = torch.cuda.max_memory_reserved(device_index) / 1024**3

    clean_latent = clean_latent.to("cpu")
    decoder_cache = cache.decoder_cache
    assert decoder_cache is not None
    del final_state, cache, text_embeddings, negative_text_embeddings
    _release_cuda_stage(diffusion)

    torch.cuda.reset_peak_memory_stats(device_index)
    decoder.to(runner.device)
    decode_start = time.perf_counter()
    generated = _decode_low_vram_chunks(
        decoder,
        clean_latent,
        decoder_cache,
        runner.device,
    )
    torch.cuda.synchronize(device_index)
    decode_seconds = time.perf_counter() - decode_start
    decode_peak_gib = torch.cuda.max_memory_allocated(device_index) / 1024**3
    decode_peak_reserved_gib = torch.cuda.max_memory_reserved(device_index) / 1024**3

    postprocess_stream = runner.create_postprocess_stream(fps=config.fps)
    postprocess_stream.process(generated, autoregressive_index=0)
    generated_cpu = postprocess_stream.finish()
    _release_cuda_stage(decoder)
    if generated_cpu is None:
        return

    artifact_name = f"{config.runner_name}-low-vram"
    stats = {
        "text_encode_seconds": text_seconds,
        "diffuse_seconds": diffuse_seconds,
        "decode_seconds": decode_seconds,
        "diffuse_peak_allocated_gib": diffuse_peak_gib,
        "diffuse_peak_reserved_gib": diffuse_peak_reserved_gib,
        "decode_peak_allocated_gib": decode_peak_gib,
        "decode_peak_reserved_gib": decode_peak_reserved_gib,
        "decode_latent_chunk_size": _LOW_VRAM_DECODE_CHUNK_SIZE,
        "gpu_memory_budget_gib": budget,
        "kv_cache_storage_gib": kv_cache_bytes / 1024**3,
        "kv_cache_uncompressed_gib": kv_cache_uncompressed_bytes / 1024**3,
        "kv_cache_tokens_per_block": kv_cache_tokens_per_block,
        "kv_cache_block_count": kv_cache_block_count,
        "kv_cache_compression_ratio": (
            kv_cache_bytes / kv_cache_uncompressed_bytes
            if kv_cache_uncompressed_bytes
            else None
        ),
    }
    ensure_output_dir(config.output_dir)
    video_path = runner_artifact_path(config.output_dir, artifact_name, "mp4")
    write_video_tensor(generated_cpu, video_path, fps=config.fps, layout="tchw")
    stats_path = write_runner_stats(
        config.output_dir,
        artifact_name,
        [{"autoregressive_index": 0, **stats}],
    )
    logger.info(
        f"[{config.runner_name}] wrote {tuple(generated_cpu.shape)} -> "
        f"{video_path.resolve()}"
    )
    logger.info(
        f"[{config.runner_name}] peaks allocated/reserved: "
        f"DiT {diffuse_peak_gib:.2f}/{diffuse_peak_reserved_gib:.2f} GiB, "
        f"VAE {decode_peak_gib:.2f}/{decode_peak_reserved_gib:.2f} GiB "
        f"(budget {budget:.2f} GiB); "
        f"stats -> {stats_path.resolve()}"
    )
