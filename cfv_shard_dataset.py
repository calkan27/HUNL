import os
import json


class CFVShardDataset:
	def __init__(
		self,
		shard_paths,
		schema_version: str = "cfv.v1",
		verify_schema: bool = True,
	):
		self.paths = list(shard_paths)
		self.schema_version = str(schema_version)
		self.verify = bool(verify_schema)

	def __iter__(self):
		for p in self.paths:
			if isinstance(p, str):
				path = p
			else:
				path = str(p)

			if os.path.isfile(path):
				if os.access(path, os.R_OK):
					with open(path, "r") as f:
						for line in f:
							if isinstance(line, str):
								txt = line.strip()
							else:
								txt = ""

							if txt:
								if txt.startswith("{"):
									if txt.endswith("}"):
										obj = json.loads(txt)

										if self.verify:
											if str(obj.get("schema", "")) == self.schema_version:
												yield obj
											else:
												print(f"[SKIP] Schema mismatch in {path}.")
										else:
											yield obj
									else:
										print(f"[SKIP] Malformed JSON (no closing brace) in {path}.")
								else:
									print(f"[SKIP] Non-JSON line in {path}.")
							else:
								print(f"[SKIP] Empty line in {path}.")
				else:
					print(f"[SKIP] No read permission for {path}.")
			else:
				print(f"[SKIP] File not found: {path}.")

