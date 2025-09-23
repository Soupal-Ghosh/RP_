# -*- coding: utf-8 -*-
"""Match_Webcam.py

Script 4 – Webcam Roll Number Matching
"""

'''
This is the main script where the matching of the roll numbers 
and the summoning of the feed will be taking place with respect to webcam
script 4
~ssmjtc
'''

"""**DEPENDENCIES**"""

import cv2
import numpy as np
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import speechrec

print("SUCCESS")

"""**MATCHING**"""

class Webcam:
    def __init__(self, embeddings_path, provider='CPUExecutionProvider'):
        # Initialize InsightFace model
        self.app = FaceAnalysis(providers=[provider])
        self.app.prepare(ctx_id=0, det_size=(480, 480))
        self.embeddings_path = embeddings_path
        self.data = self.load_embeddings(embeddings_path)

    def load_embeddings(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_embeddings_for_id(self, target_id):
        embeddings = self.data["embeddings"]
        metadata = self.data["metadata"]

        matched_embs = [
            emb for emb, meta in zip(embeddings, metadata) 
            if str(meta["id"]) == str(target_id)
        ]
        return np.array(matched_embs)

    def is_match(self, query_emb, target_embs, threshold=0.5):
        if len(target_embs) == 0:
            return False
        sims = cosine_similarity([query_emb], target_embs)[0]
        return sims.max() > (1 - threshold)

    def recognize(self, target_ids):
        """
        target_ids: list of roll numbers (integers)
        """
        # Load embeddings for all target IDs
        all_target_embs = {tid: self.get_embeddings_for_id(tid) for tid in target_ids}

        # Warn if any ID has no embeddings
        for tid, embs in all_target_embs.items():
            if len(embs) == 0:
                print(f"[WARNING] No embeddings found for ID {tid}. Available IDs:", 
                      [meta["id"] for meta in self.data["metadata"]])

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not access webcam")
            return

        print(f"[INFO] Starting recognition for IDs {target_ids}. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.app.get(frame)
            for face in faces:
                for tid, target_embs in all_target_embs.items():
                    if self.is_match(face.embedding, target_embs):
                        bbox = face.bbox.astype(int)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {tid}", (bbox[0], bbox[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Webcam Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# ---------------- Main ---------------- #
if __name__ == "__main__":
    embeddings_path = "embeddings.pkl"

    speech = speechrec.voice_to_text(duration=6)  # Record for 6 seconds
    print("Speech result:", speech, "\n")

    # Extract numbers and convert to integers
    roll_input = [int(r) for r in re.findall(r'\d+', speech)]
    print("Detected roll numbers:", roll_input)

    if not roll_input:
        print("[ERROR] No roll number detected in speech.")
    else:
        webcam_system = Webcam(embeddings_path)
        webcam_system.recognize(roll_input)
