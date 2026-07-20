"""
MinIO Presigned URL Generator + Multipart Upload

Tạo presigned URL để truy cập object trong MinIO bucket
mà không cần expose bucket ra public.

Hỗ trợ:
- Presigned GET/PUT URL (file nhỏ)
- Multipart Upload (file lớn, bypass Cloudflare 100MB limit)

Usage:
    python presign_url.py <object_path>                    # Get presigned download URL
    python presign_url.py --put <object_path>              # Get presigned upload URL
    python presign_url.py --multipart <file_path> <object_path>  # Multipart upload
    python presign_url.py --list [prefix]                  # List objects
"""

import os
import sys
import math
import hashlib
import threading
from datetime import timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from minio import Minio
from minio.error import S3Error

# ============ CẤU HÌNH ============
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "s3.lamhai.net")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "sipm-policy")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "Y9CQtFqPARVX7Zyg")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "sipm-smart-industry-management")
MINIO_SECURE = os.getenv("MINIO_SECURE", "true").lower() == "true"

# Thời gian URL hết hạn (giây)
PRESIGN_EXPIRY = int(os.getenv("PRESIGN_EXPIRY", "3600"))  # 1 giờ

# Multipart upload config
PART_SIZE = int(os.getenv("PART_SIZE", str(10 * 1024 * 1024)))  # 10MB mỗi part
MAX_CONCURRENT_UPLOADS = int(os.getenv("MAX_CONCURRENT_UPLOADS", "5"))  # Upload song song 5 parts
# ==================================


def get_client() -> Minio:
    """Tạo MinIO client."""
    return Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def generate_presigned_get_url(object_name: str, expiry_seconds: int = PRESIGN_EXPIRY) -> str:
    """
    Tạo presigned URL để GET (download) object.

    Args:
        object_name: Đường dẫn object trong bucket (vd: "uploads/avatar.png")
        expiry_seconds: Thời gian URL hết hạn (giây)

    Returns:
        Presigned URL string
    """
    client = get_client()

    url = client.presigned_get_object(
        bucket_name=MINIO_BUCKET,
        object_name=object_name,
        expires=timedelta(seconds=expiry_seconds),
    )

    return url


def generate_presigned_put_url(object_name: str, expiry_seconds: int = PRESIGN_EXPIRY) -> str:
    """
    Tạo presigned URL để PUT (upload) object.
    Frontend có thể dùng URL này để upload file trực tiếp lên MinIO.

    Args:
        object_name: Đường dẫn object muốn upload (vd: "uploads/new-image.png")
        expiry_seconds: Thời gian URL hết hạn (giây)

    Returns:
        Presigned URL string
    """
    client = get_client()

    url = client.presigned_put_object(
        bucket_name=MINIO_BUCKET,
        object_name=object_name,
        expires=timedelta(seconds=expiry_seconds),
    )

    return url


def list_objects(prefix: str = "") -> list[dict]:
    """List objects trong bucket."""
    client = get_client()

    objects = []
    for obj in client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True):
        objects.append({
            "name": obj.object_name,
            "size": obj.size,
            "last_modified": obj.last_modified,
        })

    return objects


# ============ MULTIPART UPLOAD ============

def multipart_upload(
    file_path: str,
    object_name: str,
    part_size: int = PART_SIZE,
    max_concurrent: int = MAX_CONCURRENT_UPLOADS,
) -> dict:
    """
    Upload file lớn bằng multipart upload.
    Chia file thành nhiều parts nhỏ → upload song song → MinIO ghép lại.

    Bypass Cloudflare 100MB limit vì mỗi part chỉ 5-10MB.

    Args:
        file_path: Đường dẫn file local cần upload
        object_name: Đường dẫn object trong bucket
        part_size: Kích thước mỗi part (bytes), mặc định 10MB
        max_concurrent: Số parts upload song song

    Returns:
        Dict chứa thông tin upload result
    """
    client = get_client()
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    file_size = file_path.stat().st_size
    total_parts = math.ceil(file_size / part_size)

    print(f"File: {file_path.name} ({file_size / (1024*1024):.1f} MB)")
    print(f"Parts: {total_parts} x {part_size / (1024*1024):.0f}MB")
    print(f"Concurrent uploads: {max_concurrent}")
    print()

    # Step 1: Initiate multipart upload
    upload_id = client._create_multipart_upload(
        bucket_name=MINIO_BUCKET,
        object_name=object_name,
        headers={},
    )
    print(f"Upload ID: {upload_id}")

    # Step 2: Upload parts song song
    uploaded_parts = []
    lock = threading.Lock()
    failed_parts = []

    def upload_part(part_number: int, offset: int, size: int) -> dict:
        """Upload 1 part."""
        try:
            with open(file_path, "rb") as f:
                f.seek(offset)
                data = f.read(size)

            # Tính MD5 cho part
            md5_hash = hashlib.md5(data).hexdigest()

            from io import BytesIO
            part_data = BytesIO(data)

            result = client._upload_part(
                bucket_name=MINIO_BUCKET,
                object_name=object_name,
                upload_id=upload_id,
                part_number=part_number,
                data=part_data,
                headers={
                    "Content-Length": str(size),
                },
            )

            with lock:
                uploaded_parts.append({
                    "part_number": part_number,
                    "etag": result,
                })
                progress = len(uploaded_parts) / total_parts * 100
                print(f"  Part {part_number}/{total_parts} uploaded ({progress:.0f}%)")

            return {"part_number": part_number, "etag": result}

        except Exception as e:
            with lock:
                failed_parts.append({"part_number": part_number, "error": str(e)})
            print(f"  Part {part_number} FAILED: {e}")
            return None

    # Upload song song
    print(f"\nUploading {total_parts} parts...")
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = []
        for i in range(total_parts):
            part_number = i + 1
            offset = i * part_size
            size = min(part_size, file_size - offset)
            futures.append(executor.submit(upload_part, part_number, offset, size))

        # Đợi tất cả hoàn thành
        for future in as_completed(futures):
            future.result()

    # Check failed parts
    if failed_parts:
        print(f"\n❌ {len(failed_parts)} parts failed!")
        # Abort multipart upload
        client._abort_multipart_upload(
            bucket_name=MINIO_BUCKET,
            object_name=object_name,
            upload_id=upload_id,
        )
        print("Upload aborted.")
        return {"success": False, "failed_parts": failed_parts}

    # Step 3: Complete multipart upload
    # Sort parts by part_number
    uploaded_parts.sort(key=lambda x: x["part_number"])

    client._complete_multipart_upload(
        bucket_name=MINIO_BUCKET,
        object_name=object_name,
        upload_id=upload_id,
        parts=uploaded_parts,
    )

    print(f"\n✅ Upload thành công!")
    print(f"   Object: {MINIO_BUCKET}/{object_name}")
    print(f"   Size: {file_size / (1024*1024):.1f} MB")

    return {
        "success": True,
        "object_name": object_name,
        "size": file_size,
        "parts": total_parts,
    }


def multipart_upload_simple(file_path: str, object_name: str, part_size: int = PART_SIZE) -> dict:
    """
    Upload file lớn bằng MinIO SDK built-in (fput_object tự handle multipart).
    Đơn giản hơn, MinIO SDK tự chia parts.

    Args:
        file_path: Đường dẫn file local
        object_name: Đường dẫn object trong bucket
        part_size: Kích thước mỗi part

    Returns:
        Upload result
    """
    client = get_client()
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    file_size = file_path.stat().st_size
    print(f"Uploading: {file_path.name} ({file_size / (1024*1024):.1f} MB)")
    print(f"Part size: {part_size / (1024*1024):.0f} MB")
    print()

    result = client.fput_object(
        bucket_name=MINIO_BUCKET,
        object_name=object_name,
        file_path=str(file_path),
        part_size=part_size,
    )

    print(f"✅ Upload thành công!")
    print(f"   Object: {result.bucket_name}/{result.object_name}")
    print(f"   ETag: {result.etag}")
    print(f"   Version ID: {result.version_id}")

    return {
        "success": True,
        "object_name": result.object_name,
        "etag": result.etag,
        "size": file_size,
    }


def generate_multipart_presigned_urls(
    object_name: str,
    file_size: int,
    part_size: int = PART_SIZE,
    expiry_seconds: int = PRESIGN_EXPIRY,
) -> dict:
    """
    Tạo presigned URLs cho multipart upload từ client (frontend).

    Frontend flow:
        1. Gọi API này → nhận upload_id + presigned URLs cho từng part
        2. Frontend upload từng part song song bằng PUT request
        3. Frontend gọi API complete → backend gọi CompleteMultipartUpload

    Args:
        object_name: Đường dẫn object muốn upload
        file_size: Tổng kích thước file (bytes)
        part_size: Kích thước mỗi part
        expiry_seconds: Thời gian presigned URL hết hạn

    Returns:
        Dict chứa upload_id và list presigned URLs
    """
    client = get_client()
    total_parts = math.ceil(file_size / part_size)

    # Initiate multipart upload
    upload_id = client._create_multipart_upload(
        bucket_name=MINIO_BUCKET,
        object_name=object_name,
        headers={},
    )

    # Generate presigned URL cho từng part
    presigned_urls = []
    for part_number in range(1, total_parts + 1):
        url = client.get_presigned_url(
            method="PUT",
            bucket_name=MINIO_BUCKET,
            object_name=object_name,
            expires=timedelta(seconds=expiry_seconds),
            extra_query_params={
                "partNumber": str(part_number),
                "uploadId": upload_id,
            },
        )
        presigned_urls.append({
            "part_number": part_number,
            "url": url,
        })

    return {
        "upload_id": upload_id,
        "object_name": object_name,
        "total_parts": total_parts,
        "part_size": part_size,
        "presigned_urls": presigned_urls,
    }


def complete_multipart_upload(object_name: str, upload_id: str, parts: list[dict]) -> dict:
    """
    Hoàn tất multipart upload sau khi frontend upload xong tất cả parts.

    Args:
        object_name: Đường dẫn object
        upload_id: Upload ID từ bước initiate
        parts: List dict {"part_number": int, "etag": str} từ frontend

    Returns:
        Upload result
    """
    client = get_client()

    client._complete_multipart_upload(
        bucket_name=MINIO_BUCKET,
        object_name=object_name,
        upload_id=upload_id,
        parts=parts,
    )

    return {
        "success": True,
        "object_name": object_name,
        "upload_id": upload_id,
    }


def abort_multipart_upload(object_name: str, upload_id: str):
    """Hủy multipart upload (cleanup khi fail)."""
    client = get_client()
    client._abort_multipart_upload(
        bucket_name=MINIO_BUCKET,
        object_name=object_name,
        upload_id=upload_id,
    )
    print(f"Aborted multipart upload: {upload_id}")


# ============ END MULTIPART ============


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python presign_url.py <object_path>                          # Presigned download URL")
        print("  python presign_url.py --put <object_path>                    # Presigned upload URL")
        print("  python presign_url.py --multipart <file_path> <object_path>  # Multipart upload (file lớn)")
        print("  python presign_url.py --multipart-simple <file_path> <object_path>  # Multipart (SDK built-in)")
        print("  python presign_url.py --multipart-presign <object_path> <file_size>  # Get presigned URLs cho frontend")
        print("  python presign_url.py --list [prefix]                        # List objects")
        print()
        print("Examples:")
        print("  python presign_url.py public/avatars/image.png")
        print("  python presign_url.py --put uploads/new-file.pdf")
        print("  python presign_url.py --multipart ./video.mp4 videos/meeting-2026.mp4")
        print("  python presign_url.py --multipart-presign videos/large.mp4 268435456")
        print("  python presign_url.py --list public/avatars/")
        sys.exit(1)

    try:
        if sys.argv[1] == "--list":
            prefix = sys.argv[2] if len(sys.argv) > 2 else ""
            objects = list_objects(prefix)

            if not objects:
                print("Không có object nào.")
                return

            print(f"{'Object':<60} {'Size':<12} {'Last Modified'}")
            print("-" * 100)
            for obj in objects:
                size_str = f"{obj['size'] / 1024:.1f} KB" if obj['size'] < 1024 * 1024 else f"{obj['size'] / (1024*1024):.1f} MB"
                print(f"{obj['name']:<60} {size_str:<12} {obj['last_modified']}")

        elif sys.argv[1] == "--put":
            if len(sys.argv) < 3:
                print("Error: cần chỉ định object path")
                sys.exit(1)

            object_name = sys.argv[2]
            url = generate_presigned_put_url(object_name)
            print(f"Presigned UPLOAD URL (expires in {PRESIGN_EXPIRY}s):")
            print(url)
            print()
            print("Upload bằng curl:")
            print(f'  curl -X PUT -T ./file.ext "{url}"')

        elif sys.argv[1] == "--multipart":
            if len(sys.argv) < 4:
                print("Error: cần chỉ định <file_path> <object_path>")
                print("  python presign_url.py --multipart ./video.mp4 videos/meeting.mp4")
                sys.exit(1)

            file_path = sys.argv[2]
            object_name = sys.argv[3]
            result = multipart_upload(file_path, object_name)

            if not result["success"]:
                sys.exit(1)

        elif sys.argv[1] == "--multipart-simple":
            if len(sys.argv) < 4:
                print("Error: cần chỉ định <file_path> <object_path>")
                sys.exit(1)

            file_path = sys.argv[2]
            object_name = sys.argv[3]
            multipart_upload_simple(file_path, object_name)

        elif sys.argv[1] == "--multipart-presign":
            if len(sys.argv) < 4:
                print("Error: cần chỉ định <object_path> <file_size_bytes>")
                print("  python presign_url.py --multipart-presign videos/large.mp4 268435456")
                sys.exit(1)

            object_name = sys.argv[2]
            file_size = int(sys.argv[3])
            result = generate_multipart_presigned_urls(object_name, file_size)

            print(f"Upload ID: {result['upload_id']}")
            print(f"Object: {result['object_name']}")
            print(f"Total parts: {result['total_parts']}")
            print(f"Part size: {result['part_size'] / (1024*1024):.0f} MB")
            print()
            print("Presigned URLs:")
            for part in result["presigned_urls"]:
                print(f"  Part {part['part_number']}: {part['url'][:80]}...")
            print()
            print("Frontend upload each part:")
            print('  fetch(url, { method: "PUT", body: partBlob })')
            print()
            print("After all parts uploaded, call complete with ETags:")
            print(f'  POST /api/complete-multipart')
            print(f'  Body: {{"upload_id": "{result["upload_id"]}", "object_name": "{object_name}", "parts": [{{"part_number": 1, "etag": "..."}}]}}')

        else:
            object_name = sys.argv[1]
            url = generate_presigned_get_url(object_name)
            print(f"Presigned DOWNLOAD URL (expires in {PRESIGN_EXPIRY}s):")
            print(url)

    except S3Error as e:
        print(f"MinIO Error: {e.message}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
