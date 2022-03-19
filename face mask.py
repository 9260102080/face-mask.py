from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore", category=FutureWarning)
#Function that returns the co-ordinates of faces if detected
def Face_Recognition(img):
@@ -46,23 +45,18 @@ def Face_Recognition(img):
nomask = nomask.reshape((250,50*50*3))
X = np.concatenate((mask,nomask))
labels = np.zeros(X.shape[0])
labels[100:] = 1
labels[250:] = 1
mask_nomask = {"Mask":0,"No Mask":1}
print(X.shape,labels.shape)
#Creating a Support Vector Model
svm = SVC(kernel='linear', C = 1.0)
x_train,x_test,y_train,y_test = train_test_split(X,labels,test_size = 0.3)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
#Dimentionality Reduciton(Reduction of feature variables)
pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print(x_test.shape)
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)
print(accuracy_score(y_pred,y_test))

img = cv2.imread("D:/Data/Downloads/download (1).jpg")
img = cv2.imread("Images\download (1).jpg")
faces = Face_Recognition(img)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0))
@@ -74,7 +68,5 @@ def Face_Recognition(img):
print(img.shape)
img = img.reshape((1,-1))
print(img.shape)
img = pca.transform(img)
print(img.shape)
y_pred = svm.predict(img)
print(y_pred) 