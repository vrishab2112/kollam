import os
import re


def resolve_archive_mapping(image_name: str, archive_root: str = 'archive (1)'):
    """Map an archive image filename like 'kolam29-37.jpg' to its CSV path and 1-based idx.

    Returns (csv_path, idx).
    Raises ValueError if the name doesn't match expected pattern.
    """
    m = re.match(r"kolam(\d+)-(\d+)\.(?:jpg|jpeg|png)$", image_name, re.IGNORECASE)
    if not m:
        raise ValueError(
            "Filename must look like 'kolam<set>-<k>.jpg' (e.g., kolam19-0.jpg)"
        )
    set_no = int(m.group(1))
    k = int(m.group(2))
    idx = k + 1  # dataset columns are 1-based
    csv_path = os.path.join(archive_root, 'Kolam CSV files', 'Kolam CSV files', f'kolam{set_no}.csv')
    return csv_path, idx


