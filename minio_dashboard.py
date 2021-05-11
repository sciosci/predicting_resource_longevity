from minio import Minio

MINIO_HOSTS = [
    "ist-deacuna2-s1.syr.edu:9000",
]
MINIO_USERNAME = "minio"
MINIO_PASSWORD = "SyrSosOrange!"

size_total = 0
print('')
print('Start to calculate...')
print('')
for MINIO_HOST in MINIO_HOSTS:
    size_total_host = 0
    mc = Minio(
        MINIO_HOST,
        access_key=MINIO_USERNAME,
        secret_key=MINIO_PASSWORD,
        secure=False
    )

    buckets = mc.list_buckets()

    for bucket in buckets:
        size = len(list(mc.list_objects(bucket.name)))
        print(f'Bucket Name: {bucket.name} | Size: {size}')
        size_total_host += size
        size_total += size
    print(f'Host: {MINIO_HOST}, Total: {size_total_host}')
    print('------------')
    print('')

print(f'Total size: {size_total}')
