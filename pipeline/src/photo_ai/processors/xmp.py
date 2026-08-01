"""Write enriched metadata to XMP sidecar files via ExifTool."""

from pathlib import Path
import exiftool


class XmpWriter:
    def write(
        self,
        photo_path: Path,
        caption: str | None = None,
        tags: list[str] | None = None,
        people: list[str] | None = None,
        ocr_text: str | None = None,
    ) -> None:
        xmp_path = photo_path.with_suffix(photo_path.suffix + ".xmp")
        params: list[str] = []

        if caption:
            params += [f"-XMP:Description={caption}", f"-IPTC:Caption-Abstract={caption}"]
        if tags:
            for tag in tags:
                params.append(f"-XMP:Subject+={tag}")
                params.append(f"-IPTC:Keywords+={tag}")
        if people:
            for person in people:
                params.append(f"-XMP-iptcExt:PersonInImage+={person}")
        if ocr_text:
            params.append(f"-XMP:UserComment={ocr_text}")

        if not params:
            return

        with exiftool.ExifToolHelper() as et:
            et.execute(
                *params,
                "-overwrite_original",
                f"-o={xmp_path}",
                str(photo_path),
            )
