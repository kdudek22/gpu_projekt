import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture('recordings/5/cam_2.mp4')

    ret, prev = cap.read()
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    i = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev)
        cv2.imwrite(f'recordings/5/frame_diffs/cam_2/image_{i:03d}.png', diff)
        prev, i = gray, i + 1

    cap.release()