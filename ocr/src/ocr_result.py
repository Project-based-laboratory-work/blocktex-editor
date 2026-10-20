import json
from pathlib import Path

def save_ocr_result(
    image_id,
    engine_name,
    engine_version,
    image_width,
    image_height,
    processing_time_sec,
    results,
):
    output = {
        "schema_version": "1.0",
        "image_id": image_id,
        "engine": {
            "name": engine_name,
            "version": engine_version,
        },
        "image": {
            "width": image_width,
            "height": image_height,
        },
        "processing_time_sec": round(processing_time_sec, 4),
        "results": results,
    }

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_id}_{engine_name}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output_path