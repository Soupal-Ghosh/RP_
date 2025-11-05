#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hikvision Live CCTV Face Detection + Recognition using InsightFace
Press 'Q' to quit.
"""

import cv2
import numpy as np
import pickle
import time
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from hikvisionapi import Client

class CCTV:
    def __init__(self, embeddings_path, provider='CPUExecutionProvider'):
        # Initialize InsightFace
        self.app = FaceAnalysis(providers=[provider])
        self.app.prepare(ctx_id=0, det_size=(480, 480))
        self.data = self.load_embeddings(embeddings_path)

        # Organize embeddings by student ID
        self.all_embeddings = {}
        for emb, meta in zip(self.data["embeddings"], self.data["metadata"]):
            sid = str(meta["id"])
            if sid not in self.all_embeddings:
                self.all_embeddings[sid] = []
            self.all_embeddings[sid].append(emb)
        for sid in self.all_embeddings:
            self.all_embeddings[sid] = np.array(self.all_embeddings[sid])

    def load_embeddings(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def is_match(self, query_emb, target_embs, threshold=0.5):
        if len(target_embs) == 0:
            return False
        sims = cosine_similarity([query_emb], target_embs)[0]
        return sims.max() > (1 - threshold)

# ===========================
# 🧩 Connect to the camera
# ===========================
print("Connecting to Hikvision camera...")
cam = Client('http://192.168.29.94', 'admin', 'chandrika12345', timeout=10)

# ===========================
# 🧠 Initialize Face System
# ===========================
embeddings_path = "embeddings.pkl"
cctv_system = CCTV(embeddings_path)

# ===========================
# 📷 Start Stream Loop
# ===========================
print("✅ Starting live feed... Press 'Q' to exit.")

while True:
    try:
        # Get low-res substream snapshot (Channel 102 is low-res H.264)
        response = cam.Streaming.channels[501].picture(method='get', type='opaque_data')

        # Convert bytes to image
        img_bytes = b''.join(response.iter_content(chunk_size=1024))
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("⚠️ Failed to decode frame. Retrying...")
            time.sleep(0.5)
            continue

        # Detect and recognize faces
        faces = cctv_system.app.get(frame)
        for face in faces:
            emb = face.embedding
            matched_id = None
            for sid, sid_embs in cctv_system.all_embeddings.items():
                if cctv_system.is_match(emb, sid_embs):
                    matched_id = sid
                    break

            bbox = face.bbox.astype(int)
            color = (0, 255, 0) if matched_id else (0, 0, 255)
            label = f"ID: {matched_id}" if matched_id else "Unknown"

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display window
        cv2.imshow("🔴 Live CCTV Face Recognition", frame)

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Exit requested by user.")
            break

        # Frame rate limiter (5 fps)
        time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n🛑 Keyboard Interrupt — Stopping stream.")
        break
    except Exception as e:
        print("❌ Error:", e)
        time.sleep(1)

cv2.destroyAllWindows()
print("✅ Camera disconnected. Program ended.")
