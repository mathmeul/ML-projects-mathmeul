"""security.py — bewegingsdetectie op de Iriun-camera, met lokale Moondream-analyse.

OpenCV kijkt continu naar pixelverandering. Zodra er beweging is, wordt het
frame meteen naar het lokale Moondream-model (via Ollama) gestuurd en wordt de
volledige beschrijving in de terminal geprint.

Gebruik:
    python security.py            # Iriun automatisch zoeken
    python security.py 1          # camera-index forceren
    python security.py --headless # zonder previewvenster
"""
from __future__ import annotations

import queue
import re
import sys
import threading
import time
from datetime import datetime

import cv2
import ollama

from Comms import iriun_index, open_camera, scan

MODEL = "moondream"
KEEP_ALIVE = "1h"      # model in het geheugen houden tussen detecties door

# Bewegingsdetectie
MIN_AREA = 1200        # minimale contouroppervlakte in pixels (hoger = minder gevoelig)
THRESHOLD = 25         # pixelverschil dat als beweging telt
WARMUP_FRAMES = 30     # frames om de achtergrond te laten settelen
COOLDOWN = 12.0        # min. seconden tussen twee AI-analyses; een analyse duurt op
                       # deze machine 18-30s, dus in de praktijk is dit "zo snel
                       # als het model aankan"

# Let op: dit Moondream-model (phi2) heeft "Question:" als stop-token en kapt
# ja/nee-vragen af tot een leeg antwoord. Alleen beschrijvende opdrachten
# leveren tekst op, dus we vragen nooit "is er een persoon?" maar leiden dat af
# uit een korte scenebeschrijving. Die scene-regel wordt niet geprint; hij dient
# alleen als filter.
# "Describe this image in one short sentence." leverde onzin op ("urn of water
# with a black lid") — dit model kan die beknoptheidsinstructie niet aan. De
# kale opdracht werkt wel; beknopt maken we de output zelf, met SOFT_LIMIT.
Q_GATE = "Describe this image."
Q_GATE_RETRY = "Describe this image in detail. What is happening?"
Q_APPEARANCE = (
    "Describe only the appearance of the person: the clothing they wear with "
    "colors, their hair, and their build. Do not describe the room."
)
MAX_TOKENS = 100       # harde grens; zonder limiet blijft dit model doorratelen
SOFT_LIMIT = 220       # tekens waarna we op het eerstvolgende zinseinde stoppen

# Woorden die verraden dat Moondream een mens in beeld beschrijft. Alleen dán
# vragen we door over uiterlijk — vraag je het bij een leeg beeld, dan verzint
# het model een persoon. Woordgrenzen zijn nodig: zonder \b matcht "he" in
# "The" en "her" in "other".
PERSON_PATTERN = re.compile(
    r"\b(person|persons|people|man|men|woman|women|boy|boys|girl|girls|child|"
    r"children|human|someone|somebody|individual|guy|lady|he|she|his|her|him|"
    r"they|their)\b"
)


def preload() -> None:
    """Model alvast in het geheugen laden, anders kost de eerste detectie ~10s extra."""
    try:
        ollama.generate(model=MODEL, prompt="", keep_alive=KEEP_ALIVE)
        print(f"[AI] Moondream staat geladen en klaar (keep_alive={KEEP_ALIVE}).")
    except Exception as exc:
        print(f"[WAARSCHUWING] Moondream kon niet worden voorgeladen: {exc}")
        print("[TIP] Draait Ollama? Test met: ollama run moondream")


def stream(prompt: str, image: bytes, echo: bool, soft_limit: int = 0) -> str:
    """Eén vraag aan Moondream; het antwoord verschijnt woord voor woord in de terminal."""
    chunks: list[str] = []
    started = False
    for part in ollama.generate(
        model=MODEL, prompt=prompt, images=[image], keep_alive=KEEP_ALIVE, stream=True,
        options={"num_predict": MAX_TOKENS},
    ):
        text = part["response"]
        if not started:
            # Moondream opent met witruimte; wacht op de eerste echte tekst zodat
            # het antwoord niet op een lege regel begint.
            text = text.lstrip()
            if not text:
                continue
            started = True
        chunks.append(text)
        if echo:
            print(text, end="", flush=True)

        # Netjes afronden op een zinseinde in plaats van middenin een woord
        # afgekapt worden door num_predict.
        answer = "".join(chunks)
        if soft_limit and len(answer) >= soft_limit and answer.rstrip()[-1:] in ".!?":
            break

    return "".join(chunks).strip()


def analyse(frame_bytes: bytes, detected_at: datetime) -> None:
    """Alleen melden bij beweging; beschrijven zodra het om een persoon gaat."""
    started = time.time()
    clock = detected_at.strftime("%H:%M:%S")

    try:
        # Stille filtervraag: de scene-regel zelf wordt nooit geprint, hij bepaalt
        # alleen of er een mens in beeld is.
        scene = stream(Q_GATE, frame_bytes, echo=False)
        if not scene:
            scene = stream(Q_GATE_RETRY, frame_bytes, echo=False)

        if not PERSON_PATTERN.search(scene.lower()):
            print(f"[{clock}] Beweging gedetecteerd — geen persoon ({time.time() - started:.0f}s)")
            return

        print(f"[{clock}] PERSOON: ", end="", flush=True)
        if not stream(Q_APPEARANCE, frame_bytes, echo=True, soft_limit=SOFT_LIMIT):
            print("(model gaf geen beschrijving)", end="")
        print(f"  ({time.time() - started:.0f}s)")

    except Exception as exc:
        print(f"[{clock}] [FOUT] Moondream-aanroep mislukt: {exc}")
        print("[TIP] Draait Ollama? Test met: ollama run moondream")


def analysis_worker(jobs: "queue.Queue[tuple[bytes, datetime] | None]") -> None:
    """Draait de trage AI-analyse los van de camera-loop, zodat het beeld vloeiend blijft."""
    while True:
        job = jobs.get()
        if job is None:
            return
        analyse(*job)


class MotionDetector:
    """Achtergrondmodel dat langzaam meebeweegt, zodat licht- en ruisverandering
    geen vals alarm geeft maar een binnenlopende persoon wel."""

    def __init__(self, min_area: int = MIN_AREA, threshold: int = THRESHOLD,
                 warmup: int = WARMUP_FRAMES) -> None:
        self.min_area = min_area
        self.threshold = threshold
        self.warmup = warmup
        self.background = None
        self.frames_seen = 0

    def update(self, frame) -> list:
        """Geeft de bounding boxes terug van alles wat groot genoeg beweegt."""
        self.frames_seen += 1
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        if self.background is None:
            self.background = gray.astype("float")
            return []

        cv2.accumulateWeighted(gray, self.background, 0.1)
        delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))
        mask = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.frames_seen <= self.warmup:
            return []  # achtergrond nog aan het settelen; niet alarmeren
        return [cv2.boundingRect(c) for c in contours
                if cv2.contourArea(c) >= self.min_area]


def resolve_camera(args: list[str]) -> int:
    if args and args[0].isdigit():
        return int(args[0])

    index = iriun_index()
    if index is not None:
        return index

    print("[INFO] Iriun niet op naam gevonden, camera's scannen...")
    cameras = scan()
    if not cameras:
        raise SystemExit(
            "[FOUT] Geen werkende camera gevonden — staat de Iriun Webcam Server "
            "aan op de pc en de app open op je iPhone?"
        )
    return cameras[-1][0]


def main() -> None:
    args = sys.argv[1:]
    show_window = "--headless" not in args
    index = resolve_camera([a for a in args if not a.startswith("--")])

    camera = open_camera(index)
    if camera is None:
        raise SystemExit(f"[FOUT] Kan camera-index {index} niet openen.")

    preload()
    print(f"[INFO] Bewaking gestart op camera-index {index}.")
    print("[INFO] OpenCV scant op beweging; Moondream slaapt tot er iets gebeurt.")
    if show_window:
        print("[INFO] Druk op 'q' in het venster om te stoppen.")

    jobs: "queue.Queue[tuple[bytes, datetime] | None]" = queue.Queue(maxsize=1)
    worker = threading.Thread(target=analysis_worker, args=(jobs,), daemon=True)
    worker.start()

    detector = MotionDetector()
    last_analysis = 0.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("[FOUT] Frame grab mislukt — controleer de Iriun-verbinding.")
                break

            boxes = detector.update(frame)
            now = time.time()

            if boxes and now - last_analysis > COOLDOWN:
                # Het schone frame sturen, vóór de rode kaders erop staan — anders
                # gaat Moondream die rechthoeken beschrijven.
                ok_jpeg, buffer = cv2.imencode(".jpg", frame)
                if ok_jpeg:
                    try:
                        jobs.put_nowait((buffer.tobytes(), datetime.now()))
                        last_analysis = now
                    except queue.Full:
                        pass  # vorige analyse loopt nog; deze beweging overslaan

            if show_window:
                for x, y, w, h in boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                status = "BEWEGING" if boxes else "rustig"
                colour = (0, 0, 255) if boxes else (0, 200, 0)
                cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
                cv2.imshow("Bewaking (q = stoppen)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            jobs.put_nowait(None)  # blokkerend maken zou het afsluiten laten wachten
        except queue.Full:
            pass                   # worker is een daemon en verdwijnt met het proces
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Bewaking gestopt.")


if __name__ == "__main__":
    main()
