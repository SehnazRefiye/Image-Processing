import cv2
import numpy as np
from matplotlib import pyplot as py
from imutils import perspective
from imutils import contours
import imutils

# görüntü okundu renkli görüntü olduğu için 1
RGB = cv2.imread("karakter.JPG", 1)
# cv2.imshow("Original Image", RGB)

# görüntü kopyalandı
RGB_copy = RGB.copy()
RGB_copy1 = RGB.copy()

# görüntü griye çevrildi
I_gray = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray Image", I_gray)

# görüntü binary yani siyah beyaz hale getirildi
# eşik değerinin üstünde kalanlar siyah, altında kalanlar beyaz
# otsu methodu otomatik olarak bir eşik değeri bulur daha sonra görüntünün en ideal şekilde bu eşikten geçirilmesini sağlar
rev1, I_thresh = cv2.threshold(I_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imshow("Black and White Image", I_thresh)

"""
if cv2.waitKey(0) & 0xff == ord('c'):
    cv2.destroyAllWindows()
"""
# eşikten geçirilmiş siyah-beyaz görüntüyü gürültü etkisinden kurtarmak için karakterlerin kapladığı piksel alanın altında kalanları siler
def bwareaopen(image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # piksel alanı 10'un aşağısında olanı sıfırla
    min_size = 10
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

I_bw = bwareaopen(I_thresh)
I_bn = I_bw.astype(np.uint8)
# cv2.imshow("After noise reduction in black and white image", I_bw)

# kenar bilgileri bulundu, ağaç yapısı kullanıldı,
# herbir sütunda arka plandan farkı bir piksel arıyor ve bulduğu yerleri etiketliyor
I_contours, _ = cv2.findContours(I_bn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# etiketledikleri görüntüdeki birer nesne, her nesnenin etrafını çiziyor
I_drawcountours = cv2.drawContours(RGB_copy1, I_contours, -1, (0, 255, 0), 1)
# cv2.imshow("Drawcountours Image", I_dilation)

# yapısal element oluşturuldu, 2x2 tipinde, 0-255 arasında değere sahip
strel = np.ones((2, 2), np.uint8)
# görüntüyü genişletiliyor, işlem 1 kere yapılıyor
I_dilation = cv2.dilate(I_bn, strel, iterations=1)
# cv2.imshow("Dilation Image", I_dilation)

# genişletilmiş görüntü kopyalandı
I_fill = I_dilation.copy()


h, w = I_fill.shape[:2]
# genişletilmiş görüntünün uzunluğunda ve genişliğinde bir siyah görüntü oluşturuldu
mask = np.zeros((h + 2, w + 2), np.uint8)

I_fill = I_fill.astype(np.uint8)
cv2.floodFill(I_fill, mask, (0, 0), 255)

# beyaz içerisinde kalan siyah alanlar beyaza çevrildi
I_fill_inv = cv2.bitwise_not(I_fill)
# cv2.imshow("The blacks inside the white areas changed to white and only those are displayed", I_fill_inv)

# genişletilmiş görintü ve doldurulmuş görüntünün tersi birleştirildi
I_out = I_dilation | I_fill_inv
# cv2.imshow("Blacks inside the white areas became white", I_out)

# doldurma işlemi yapıldıktan sonra tekrar kenar bilgileri bulunuyor
cnts = cv2.findContours(I_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# kopyalanmış orijinal görüntü bir değişkene atandı
orig = RGB.copy()
# cv2.imshow("Original Image", orig)

# en küçük kare içerisine alma işlemi
def mid_point(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# loop over the contours individually
def boundingBox(cnts):
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 10:
            continue
        # compute the rotated bounding box of the contour
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left order,
        # then draw the outline of the rotated bounding box
        # Bounding box `(min_row, min_col, max_row, max_col)`
        box = perspective.order_points(box)
        m = cv2.drawContours(orig, [box.astype('int')], -1, (0, 255, 0), 2)
    return m

""" 
    # Görsel olarak göstermek için
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 0.955

        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        cv2.putText(orig, "{:.1f}in".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.15, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}in".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.15, (255, 255, 255), 2)
"""
def ellipse():
    for i in cnts:
        ellipse = cv2.fitEllipse(i)
        cv2.ellipse(RGB_copy, ellipse, (0, 255, 0), 2)

# herbir bounding box içerisinde kalan piksellerin sayısı
def area(cnts):
    i = 0
    for c in cnts:
        i += 1
        area = cv2.contourArea(c)
        print("Area " + str(i) + ":", str(area))


def majorAxisLength():
    j = 0
    for i in cnts:
        j += 1
        if len(i) >= 5:
            ellipse = cv2.fitEllipse(i)
            cv2.ellipse(RGB_copy, ellipse, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(i)
            if w > h:
                majorAxisLength = w / 2
            else:
                majorAxisLength = h / 2
            print("Major Axis Length " + str(j) + ":", str(majorAxisLength))

def minorAxisLength():
    j = 0
    for i in cnts:
        j += 1
        if len(i) >= 5:
            ellipse = cv2.fitEllipse(i)
            cv2.ellipse(RGB_copy, ellipse, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(i)
            if w < h:
                minorAxisLength = w / 2
            else:
                minorAxisLength = h / 2
            print("Minor Axis Length " + str(j) + ":", str(minorAxisLength))

# eccentricity = sqrt( 1 - (ma/MA)^2) --- ma= minor axis --- MA= major axis
def eccentricity():
    j = 0
    for i in cnts:
        j += 1
        if len(i) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(i)
            a = ma / 2
            b = MA / 2
            ecc = np.sqrt(a ** 2 - b ** 2) / a
            print("Eccentricity " + str(j) + ":", str(ecc))


orig = boundingBox(cnts)
area(cnts)
print()
majorAxisLength()
print()
minorAxisLength()
print()
eccentricity()


# yazdırma işlemi
py.subplot(331), py.imshow(RGB, cmap='gray', interpolation='nearest'), py.title('Original Image')
py.subplot(332), py.imshow(I_gray, cmap='gray', interpolation='nearest'), py.title('Gray Image')
py.subplot(333), py.imshow(I_thresh, cmap='gray', interpolation='nearest'), py.title('Binary Threshold')
py.subplot(334), py.imshow(I_bw, cmap='gray', interpolation='nearest'), py.title('Noise Reduced Image')
py.subplot(335), py.imshow(I_drawcountours, cmap='gray', interpolation='nearest'), py.title('Drawn Around Objects in the Image')
py.subplot(336), py.imshow(I_dilation, cmap='gray', interpolation='nearest'), py.title('Dilation Image')
py.subplot(337), py.imshow(I_out, cmap='gray', interpolation='nearest'), py.title('(Dilation and Fill) Image')
py.subplot(338), py.imshow(orig, cmap='gray', interpolation='nearest'), py.title('Bounding Box')
py.subplot(339), py.imshow(RGB_copy, cmap='gray', interpolation='nearest'), py.title('Ellipse')
py.show()
