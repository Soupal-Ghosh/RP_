# -*- coding: utf-8 -*-
"""
Matching_recorded.py (converted for VS Code from Colab)
"""

import cv2
import numpy as np
import pickle
import threading
import queue
import time
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import random
import speechrec
print("SUCCESS")

"""**FETCHING EMBEDD**"""

embeddings_path = "embeddings.pkl"

with open(embeddings_path, "rb") as f:
    data = pickle.load(f)

print(type(data))
print(len(data))
print(data.keys())  # or list(data.keys()) if it's a dict

"""**Recorded Video Matching**"""

class LiveFaceMatcherMulti:
    def __init__(self, embeddings_path, provider='CPUExecutionProvider'):
        self.app = FaceAnalysis(providers=[provider])
        self.app.prepare(ctx_id=0, det_size=(480, 480))
        self.data = self.load_embeddings(embeddings_path)

    def load_embeddings(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_embeddings_for_id(self, target_id):
        embeddings = self.data["embeddings"]
        metadata = self.data["metadata"]
        return np.array([emb for emb, meta in zip(embeddings, metadata) if str(meta["id"]) == str(target_id)])

    def is_match(self, query_emb, target_embs, threshold=0.5):
        if len(target_embs) == 0:
            return False
        sims = cosine_similarity([query_emb], target_embs)[0]
        return sims.max() > (1 - threshold)

    def process_video_live_multi(self, video_path, target_ids, output_path="processed_output.mp4"):
        target_embs_dict = {tid: self.get_embeddings_for_id(tid) for tid in target_ids}
        status = {tid: False for tid in target_ids}

        colors = {tid: tuple(np.random.randint(0, 256, 3).tolist()) for tid in target_ids}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Saving processed video to: {os.path.abspath(output_path)}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.app.get(frame)
            for face in faces:
                emb = face.embedding
                for tid, target_embs in target_embs_dict.items():
                    if self.is_match(emb, target_embs):
                        bbox = face.bbox.astype(int)
                        color = colors[tid]
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(frame, f"ID: {tid}", (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        status[tid] = True

            # Status bar
            y0, dy = 30, 25
            for i, tid in enumerate(target_ids):
                text = f"ID {tid}: {'Present' if status[tid] else 'Absent'}"
                color = colors[tid] if status[tid] else (0, 0, 255)
                cv2.putText(frame, text, (10, y0 + i*dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Show live video in VS Code window
            cv2.imshow("Processed Video", frame)

            # Save frame
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print("\n--- Detection Status ---")
        for tid, present in status.items():
            print(f"ID {tid}: {'Present' if present else 'Absent'}")


# ---------------- Main ---------------- #
video_path = "test2_cctv.mp4"
embeddings_path = "embeddings.pkl"

speech = speechrec.voice_to_text(duration=6)  # Record for 6 seconds
print(speech+"\n")
roll_input = re.findall(r'\d+', speech)  # Find all sequences of digits
target_ids = [int(r) for r in roll_input]  # Convert to integers

matcher = LiveFaceMatcherMulti(embeddings_path)
matcher.process_video_live_multi(video_path, target_ids, output_path="processed_output.mp4")
