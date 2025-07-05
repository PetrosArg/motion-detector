import cv2

# Άνοιγμα κάμερας
cap = cv2.VideoCapture(0)

# Παίρνουμε το πρώτο frame ως background για σύγκριση
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Υπολογισμός διαφοράς μεταξύ των 2 frames
    diff = cv2.absdiff(frame1, frame2)

    # Μετατροπή σε γκρι κλίμακα (ειναι πιο εύκολο να επεξεργαστεί)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Θόλωση για μείωση θορύβου
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Thresholding για να πάρουμε μόνο τις μεγάλες αλλαγές (τιμες πανω από 20 γινονται λευκές)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate για να ενισχύσουμε τα όρια
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Βρες τα περιγράμματα
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        # Σχεδίασε ορθογώνιο γύρω από τα αντικείμενα που κινούνται
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame1, "Motion Detected", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # Εμφάνιση εικόνας
    cv2.imshow("Motion Detector", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    # Πάτα q για έξοδο
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()