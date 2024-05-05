# importing required library
from __future__ import annotations

from prettytable import PrettyTable


def get_model_lookup_table():
    # creating an empty PrettyTable
    x = PrettyTable()

    # adding data into the table
    # row by row
    x.field_names = ["Model", "Description", "Samples"]
    x.add_row(
        [
            "1x_ArtClarity.pth",
            "Texture retaining denoiser and sharpener for digital artwork",
            "n/a",
        ]
    )
    x.add_row(
        [
            "1x_ArtClarity_strong.pth",
            "Texture retaining denoiser and sharpener for digital artwork",
            "n/a",
        ]
    )
    x.add_row(
        [
            "1x_DeSharpen.pth",
            "Made for rare particular cases when the image was destroyed by applying noise, i.e. game textures or any badly exported photos. If your image does not have any oversharpening, it won't hurt them, leaving as is. In theory, this model knows when to activate and when to skip, also can successfully remove artifacts if only some parts of the image are oversharpened, for example in image consisting of several combined images, 1 of them with sharpen noise.",
            "n/a",
        ]
    )
    x.add_row(["1x_NoiseToner-Poisson-Detailed_108000_G.pth", "Noise remover", "n/a"])
    x.add_row(["1x_NoiseToner-Poisson-Soft_101000_G.pth", "Noise remover", "n/a"])
    x.add_row(["1x_NoiseToner-Uniform-Detailed_100000_G.pth", "Noise remover", "n/a"])
    x.add_row(["1x_NoiseToner-Uniform-Soft_100000_G.pth", "Noise remover", "n/a"])
    x.add_row(["1x_NoiseTonerV1_110000_G.pth", "Noise remover", "n/a"])
    x.add_row(["1x_NoiseTonerV2_105000_G.pth", "Noise remover", "n/a"])
    x.add_row(["1x_NoiseToner_Poisson_150000_G.pth", "Noise remover", "n/a"])
    x.add_row(["1x_NoiseToner_Uniform_100000_G.pth", "Noise remover", "n/a"])
    x.add_row(["1x_PixelSharpen_v2.pth", "Restores blurry/upscaled pixel art.", "n/a"])
    x.add_row(["1x_PixelSharpen_v2_strong.pth", "Restores blurry/upscaled pixel art.", "n/a"])
    x.add_row(["1x_ReContrast.pth", "n/a", "n/a"])
    x.add_row(
        [
            "2x_KemonoScale_v2.pth",
            "Anime. Upscaling frames from Irodori anime (namely kemono friends) from 540p (the source render resolution) to 1080p, low resolution flat shaded art, de-JPEG of the aforementioned",
            "n/a",
        ]
    )
    x.add_row(
        [
            "2x_MangaScaleV3.pth",
            "To upscale manga including halftones, instead of trying to smooth them out.",
            "n/a",
        ]
    )
    x.add_row(
        [
            "4x-UltraSharp.pth",
            "Universal Upscaler. This is my best model yet! It generates lots and lots of detail and leaves a nice texture on images. It works on most images, whether compressed or not. It does work best on JPEG compression though, as that's mostly what it was trained on. It has the ability to restore highly compressed images as well!",
            "n/a",
        ]
    )
    x.add_row(
        [
            "4xFSMangaV2.pth",
            "Manga-style images with or without dithering - cartoons, maybe pixel art, etc	",
            "n/a",
        ]
    )
    x.add_row(["4x_BigFace_v3.pth", "Art/People", "n/a"])
    x.add_row(["4x_BigFace_v3_Blend.pth", "Art/People", "n/a"])
    x.add_row(["4x_BigFace_v3_Clear.pth", "Art/People", "n/a"])
    x.add_row(
        [
            "4x_BooruGan_600k.pth",
            "Anime This model is designed to mainly upscale anime artworks. If you have issues with chroma then try the 600k iterations release.",
            "n/a",
        ]
    )
    x.add_row(
        [
            "4x_BooruGan_650k.pth",
            "Anime This model is designed to mainly upscale anime artworks. If you have issues with chroma then try the 600k iterations release.",
            "n/a",
        ]
    )
    x.add_row(
        [
            "4x_CountryRoads_377600_G.pth",
            "Universal Upscaler. Streets with dense foliage in the background. Outdoor scenes.",
            "n/a",
        ]
    )
    x.add_row(
        [
            "4x_FArtDIV3_Base.pth",
            "Art. Painting style with larger shaped features",
            "n/a",
        ]
    )
    x.add_row(
        [
            "4x_FArtDIV3_Blend.pth",
            "Art. Painting style with larger shaped features",
            "n/a",
        ]
    )
    x.add_row(
        [
            "4x_FArtDIV3_Fine.pth",
            "Art. Painting style with larger shaped features",
            "n/a",
        ]
    )
    x.add_row(
        [
            "4x_FArtDIV3_UltraMix4.pth",
            "Art. Painting style with larger shaped features",
            "n/a",
        ]
    )
    x.add_row(["4x_FArtFace.pth", "Art. Painting style with larger shaped features", "n/a"])
    x.add_row(
        [
            "4x_FArtSuperBlend.pth",
            "Art. Painting style with larger shaped features",
            "n/a",
        ]
    )
    x.add_row(
        [
            "4x_FatalPixels_340000_G.pth",
            "Pixel Art/Sprites. Dataset. Anime, Manga",
            "n/a",
        ]
    )
    x.add_row(
        [
            "4x_Fatality_Faces_310000_G.pth",
            "Pixel Art/Sprites. Upscales medium resolution Sprites, dithered or undithered, can also upscale manga/anime and gameboy camera images.",
            "n/a",
        ]
    )
    x.add_row(["4x_Unholy_FArt.pth", "description", "n/a"])
    x.add_row(
        [
            "4x_Valar_v1.pth",
            "Realistic Photos. Meant as an experiment to test latest techniques implemented on traiNNer, including: AdaTarget, KernelGAN, UNet discriminator, nESRGAN+ arch, noise patches, camera noise, isotropic/anisotropic/sinc blur, frequency separation, contextual loss, mixup, clipL1 pixel loss, AdamP optimizer, etc. The config file is provided on the download link above. I encourage everybody to mirror the model, distribute and modify it in anywway you want.",
            "n/a",
        ]
    )
    x.add_row(
        [
            "4x_detoon_225k.pth",
            "Pixel art/sprites. For upscaling character sprites",
            "n/a",
        ]
    )
    x.add_row(
        [
            "8x_BoyMeBob-Redux_200000_G.pth",
            "Upscaling cartoons. ESRGAN+ (Joey's fork, eFonte fork, or iNNfer required to use)",
            "n/a",
        ]
    )
    x.add_row(
        [
            "8x_NMKD-Typescale_175k.pth",
            "Text. Low-resolution text/typography and symbols",
            "n/a",
        ]
    )
    x.add_row(["KemonoClean.pth", "n/a", "n/a"])
    x.add_row(
        [
            "LADDIER1_282500_G.pth",
            "Denoise. Remove noise, grain, box blur, lens blur and gaussian blur and increase overall image quality.",
            "n/a",
        ]
    )
    x.add_row(
        [
            "RRDB_ESRGAN_x4.pth",
            "Game textures. Various game textures. Primary wood, metal, stone",
            "n/a",
        ]
    )
    x.add_row(
        [
            "RRDB_ESRGAN_x4_old_arch.pth",
            "Game textures. Various game textures. Primary wood, metal, stone",
            "n/a",
        ]
    )
    x.add_row(
        [
            "RRDB_PSNR_x4.pth",
            "pretrained model. The original RRDB_PSNR_x4.pth model converted to 1x, 2x, 8x and 16x scales, intended to be used as pretrained models for new models at those scales. These are compatible with victor's 4xESRGAN.pth conversions",
            "n/a",
        ]
    )
    x.add_row(
        [
            "RRDB_PSNR_x4_old_arch.pth",
            "pretrained model. The original RRDB_PSNR_x4.pth model converted to 1x, 2x, 8x and 16x scales, intended to be used as pretrained models for new models at those scales. These are compatible with victor's 4xESRGAN.pth conversions",
            "n/a",
        ]
    )
    x.add_row(["RealESRGANv2-animevideo-xsx2.pth", "anime", "n/a"])
    x.add_row(["RealESRGANv2-animevideo-xsx4.pth", "anime", "n/a"])
    x.add_row(["TGHQFace8x_500k.pth", "Painted humans", "n/a"])
    x.add_row(["detoon_alt.pth", "cartoons", "n/a"])
    x.add_row(["furry_12400_G.pth", "description", "n/a"])

    # printing generated table
    print(x)
    return x
