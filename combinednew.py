#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hikvision Multi-Camera Live CCTV Face Detection + Recognition + Voice Search
Continuous Search Mode - Fixes for Embeddings Matching
"""

import cv2
import numpy as np
import pickle
import time
import re
import speech_recognition as sr
import pyttsx3
import threading
import queue
from collections import deque
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from hikvisionapi import Client

# ============================================================
# 🎤 Voice Engine
# ============================================================
engine = pyttsx3.init()
engine.setProperty('rate', 165)
engine.setProperty('voice', engine.getProperty('voices')[1].id)

def speak(text):
    print(f"🗣️ {text}")
    engine.say(text)
    engine.runAndWait()

def listen_for_command():
    r = sr.Recognizer()
    r.energy_threshold = 4000
    r.dynamic_energy_threshold = True
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("🎧 Listening for command... (say 'Search ID 1')")
        try:
            audio = r.listen(source, timeout=3, phrase_time_limit=4)
        except sr.WaitTimeoutError:
            return None

    number_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12
    }

    try:
        command = r.recognize_google(audio).lower()
        print("🎤 You said:", command)

        # Extract numeric ID
        match = re.search(r'id[- ]?(\d+)', command)
        if match:
            return int(match.group(1))

        # Extract word-based ID
        for word, num in number_words.items():
            if f"id {word}" in command or f"search id {word}" in command:
                return num

        return None

    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        speak("Speech recognition error.")
        return None

# ============================================================
# 🧠 Face Recognition - FIXED VERSION
# ============================================================
class CCTV:
    def __init__(self, embeddings_path, provider='CPUExecutionProvider'):
        # Use same det_size as your original code for consistent embeddings
        self.app = FaceAnalysis(providers=[provider])
        self.app.prepare(ctx_id=0, det_size=(480, 480))  # IMPORTANT: Same as when embeddings were created
        self.data = self.load_embeddings(embeddings_path)

        # Group embeddings by ID
        self.all_embeddings = {}
        for emb, meta in zip(self.data["embeddings"], self.data["metadata"]):
            sid = str(meta["id"])
            if sid not in self.all_embeddings:
                self.all_embeddings[sid] = []
            self.all_embeddings[sid].append(emb)

        for sid in self.all_embeddings:
            self.all_embeddings[sid] = np.array(self.all_embeddings[sid])
        
        print(f"✅ Loaded embeddings for IDs: {list(self.all_embeddings.keys())}")
        print(f"   Total embeddings: {sum(len(embs) for embs in self.all_embeddings.values())}")

    def load_embeddings(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def is_match(self, query_emb, target_embs, threshold=0.5):
        if len(target_embs) == 0:
            return False
        sims = cosine_similarity([query_emb], target_embs)[0]
        max_sim = sims.max()
        
        # Debug: Print similarity scores
        if max_sim > 0.5:  # Only print high similarities for debugging
            print(f"   Similarity: {max_sim:.3f} (threshold: {1-threshold})")
        
        return max_sim > (1 - threshold)

    def recognize_face(self, query_emb):
        """Return recognized ID or None"""
        best_id = None
        best_similarity = 0
        
        for sid, sid_embs in self.all_embeddings.items():
            if len(sid_embs) == 0:
                continue
                
            sims = cosine_similarity([query_emb], sid_embs)[0]
            max_sim = sims.max()
            
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_id = sid
        
        # Use threshold 0.5 (similarity > 0.5)
        return best_id if best_similarity > 0.5 else None

# ============================================================
# 📷 Camera Streamer
# ============================================================
class CameraStreamer:
    def __init__(self, cam_client, channel, cam_index):
        self.cam = cam_client
        self.channel = channel
        self.cam_index = cam_index
        self.frame_queue = deque(maxlen=2)
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.active = False
        
    def start(self):
        self.thread.start()
        # Wait for activation
        for _ in range(10):
            if self.active:
                return self
            time.sleep(0.1)
        return self
        
    def _stream_worker(self):
        while self.running:
            try:
                # Get frame from camera
                response = self.cam.Streaming.channels[self.channel].picture(
                    method='get',
                    type='opaque_data'
                )
                
                # Read image data
                img_bytes = bytearray()
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        break
                    img_bytes.extend(chunk)
                    if len(img_bytes) > 100000:  # Enough for a good image
                        break
                    
                if len(img_bytes) < 5000:
                    time.sleep(0.1)
                    continue
                    
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Keep original size for better face detection
                    with self.lock:
                        self.frame_queue.clear()
                        self.frame_queue.append(frame)
                        if not self.active:
                            self.active = True
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Camera {self.channel} error: {str(e)[:50]}")
                time.sleep(1)
                
    def get_frame(self):
        with self.lock:
            if self.frame_queue:
                return self.frame_queue[-1].copy()
        return None
        
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)

# ============================================================
# 🎯 Main Application - Continuous Search Mode
# ============================================================
def main():
    print("\n" + "="*50)
    print("HIKVISION FACE RECOGNITION - CONTINUOUS SEARCH MODE")
    print("="*50)
    
    ip = input("Enter camera IP address (e.g., 192.168.29.94): ").strip()
    username = input("Enter camera username: ").strip()
    password = input("Enter camera password: ").strip()
    num_cameras = int(input("How many cameras? ").strip())

    # Generate channel numbers
    camera_channels = [(i * 100) + 1 for i in range(1, num_cameras + 1)]
    print(f"\n📸 Camera channels: {camera_channels}")

    print("\nConnecting to Hikvision NVR/DVR...")
    try:
        cam = Client(f'http://{ip}', username, password)
        print("✅ NVR/DVR connection successful")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return

    embeddings_path = "embeddings.pkl"
    try:
        cctv_system = CCTV(embeddings_path)
        print("✅ Face recognition system loaded")
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return

    # Start camera streamers
    print("\nStarting camera streams...")
    streamers = []
    
    for i, channel in enumerate(camera_channels):
        print(f"Starting Camera {i+1} on channel {channel}...", end=" ", flush=True)
        try:
            streamer = CameraStreamer(cam, channel, i).start()
            if streamer.active:
                streamers.append((f"Camera{i+1}", streamer, channel))
                print(f"✅")
            else:
                print(f"❌")
        except Exception as e:
            print(f"❌ (Error: {str(e)[:30]})")
        time.sleep(0.5)
    
    if not streamers:
        print("❌ No cameras could be started")
        return

    print(f"\n✅ System Ready with {len(streamers)} cameras.")
    print("   Say 'Search ID 1' to begin continuous search")
    print("   Say 'Stop Search' to stop current search")
    print("   Press Q to quit | S for manual search")

    current_search_id = None
    search_active = False
    last_found_time = None
    found_in_camera = None
    last_listen_time = time.time()
    
    # Create windows
    for cam_name, _, _ in streamers:
        cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cam_name, 640, 480)

    while True:
        try:
            # Listen for commands periodically
            current_time = time.time()
            if current_time - last_listen_time > 2:  # Check every 2 seconds
                command_id = listen_for_command()
                if command_id is not None:
                    if command_id == 0:  # "stop" or similar
                        if search_active:
                            speak(f"Stopping search for ID {current_search_id}.")
                            search_active = False
                            current_search_id = None
                    else:
                        current_search_id = command_id
                        search_active = True
                        speak(f"Now continuously searching for ID {current_search_id}. Say 'Stop Search' to stop.")
                        found_in_camera = None
                        last_found_time = None
                last_listen_time = current_time

            # Process all cameras
            for cam_name, streamer, channel in streamers:
                frame = streamer.get_frame()
                
                if frame is None:
                    # Create placeholder
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, f"{cam_name} - No Signal", (50, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow(cam_name, frame)
                    continue
                
                # Make a copy for display
                display_frame = frame.copy()
                
                # Face detection (always run, but only match if searching)
                faces = cctv_system.app.get(frame)
                
                for face in faces:
                    emb = face.embedding
                    bbox = face.bbox.astype(int)
                    
                    # Always recognize faces (for display)
                    recognized_id = cctv_system.recognize_face(emb)
                    
                    if recognized_id:
                        # Known person
                        color = (0, 255, 0)  # Green
                        label = f"ID: {recognized_id}"
                        
                        # Check if this is our search target
                        if search_active and str(recognized_id) == str(current_search_id):
                            if found_in_camera != cam_name:
                                found_in_camera = cam_name
                                last_found_time = time.strftime("%H:%M:%S")
                                speak(f"ID {current_search_id} found in {cam_name} at {last_found_time}.")
                            # Highlight target with thicker border
                            cv2.rectangle(display_frame, (bbox[0], bbox[1]),
                                        (bbox[2], bbox[3]), (0, 255, 255), 4)  # Yellow highlight
                            label = f"TARGET: ID {recognized_id}"
                            cv2.putText(display_frame, "TARGET", (bbox[0], bbox[1] - 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    else:
                        # Unknown person
                        color = (0, 0, 255)  # Red
                        label = "Unknown"
                    
                    # Draw face box
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]),
                                (bbox[2], bbox[3]), color, 2)
                    cv2.putText(display_frame, label, (bbox[0], bbox[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add status overlay
                status_y = 30
                if search_active:
                    status_text = f"SEARCHING FOR: ID {current_search_id}"
                    status_color = (0, 255, 255)  # Yellow
                    
                    if found_in_camera == cam_name and last_found_time:
                        status_text += f" | FOUND HERE at {last_found_time}"
                        status_color = (0, 255, 0)  # Green
                    
                    cv2.putText(display_frame, status_text, (10, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                else:
                    if current_search_id:
                        cv2.putText(display_frame, f"LAST SEARCH: ID {current_search_id} (Stopped)", 
                                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "READY - Say 'Search ID X' to begin", 
                                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show FPS/camera info
                cv2.putText(display_frame, cam_name, (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display
                cv2.imshow(cam_name, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                speak("System shutting down. Goodbye.")
                break
            elif key == ord('s'):  # Manual search
                try:
                    search_id = int(input("\nEnter ID to search (0 to stop): "))
                    if search_id == 0:
                        if search_active:
                            speak(f"Stopping search for ID {current_search_id}.")
                        search_active = False
                        current_search_id = None
                    else:
                        current_search_id = search_id
                        search_active = True
                        speak(f"Now searching for ID {current_search_id}.")
                        found_in_camera = None
                except:
                    pass
            elif key == ord('r'):  # Reset/stop search
                if search_active:
                    speak(f"Stopping search for ID {current_search_id}.")
                search_active = False
                current_search_id = None
                found_in_camera = None
            elif key == ord('d'):  # Debug: Print embeddings info
                print(f"\n📊 Embeddings Info:")
                print(f"   Loaded IDs: {list(cctv_system.all_embeddings.keys())}")
                for sid, embs in cctv_system.all_embeddings.items():
                    print(f"   ID {sid}: {len(embs)} embeddings")
            
            # Small delay
            time.sleep(0.01)

        except KeyboardInterrupt:
            speak("System stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)

    # Cleanup
    print("\nStopping all camera streams...")
    for _, streamer, _ in streamers:
        streamer.stop()
    cv2.destroyAllWindows()
    print("✅ Program ended.")

if __name__ == "__main__":
    main()