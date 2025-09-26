# # api/server.py
# import io
# import numpy as np
# import cv2
# from fastapi import FastAPI, File, UploadFile, Form
# from pydantic import BaseModel
# from detection.scrfd_detector import SCRFDDetector
# from recognition.database import FaceDatabase

# app = FastAPI()
# detector = SCRFDDetector(ctx_id=0)
# db = FaceDatabase()

# class RecognizeResponse(BaseModel):
#     names: list
#     boxes: list
#     scores: list

# @app.post("/recognize/", response_model=RecognizeResponse)
# async def recognize_image(file: UploadFile = File(...)):
#     data = await file.read()
#     nparr = np.frombuffer(data, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     faces = detector.detect(img)
#     names, boxes, scores = [], [], []
#     for f in faces:
#         name, sim = db.match(f["embedding"])
#         names.append(name)
#         boxes.append(f["bbox"])
#         scores.append(sim)
#     return RecognizeResponse(names=names, boxes=boxes, scores=scores)

# @app.post("/add_person/")
# async def add_person(name: str = Form(...), files: list[UploadFile] = File(...)):
#     embeddings = []
#     for file in files:
#         data = await file.read()
#         nparr = np.frombuffer(data, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         faces = detector.detect(img)
#         if len(faces) == 0:
#             continue
#         # take largest face
#         faces = sorted(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), reverse=True)
#         emb = faces[0]['embedding']
#         if emb is not None:
#             embeddings.append(emb)
#     if len(embeddings) == 0:
#         return {"ok": False, "msg": "no faces found in uploaded images"}
#     db.save_person(name, embeddings)
#     return {"ok": True, "msg": f"Saved {len(embeddings)} embeddings for {name}"}
