from datasets import load_dataset

ds = load_dataset("uitnlp/vigoemotions")
print(ds)
for split in ds:
    print(split, len(ds[split]), ds[split].column_names)
    print(ds[split].features)

# một dòng, cắt ngắn văn bản
row = ds["train"][0]
print({k: (v[:60] if isinstance(v, str) else v) for k, v in row.items()})