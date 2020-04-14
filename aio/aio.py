
# GUI
# voice



from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QDate, QTime, QDateTime, Qt
import sys
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication
import mysql.connector
from mysql.connector import Error
import speech_recognition as sr
from PIL import Image
import imagehash
import cv2 as cv

now = QDate.currentDate()

r = sr.Recognizer()

def voice_command():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("SAY Next or Previous")
        audio = r.listen(source)
                    
    try:
        a = r.recognize_google(audio)

    except:
        pass


    # Create a black image
    img = np.zeros((512,512,3), np.uint8)

    # Write some Text

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    blc = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(img,a,
    blc,
    font,
    fontScale,
    fontColor,
    lineType)


    #Display the image
    cv2.imshow("img",img)

    #Save image
    cv2.imwrite("out.jpg", img)




    image = cv2.imread("saved.jpg",0)
    height,width=image.shape[:2]
    WP1= cv2.countNonZero(image)
    print(WP1)

    image = cv2.imread("out.jpg",0)
    height,width=image.shape[:2]
    WP2= cv2.countNonZero(image)
    print(WP2)

    X=WP1-WP2
    print(X)

    if X == 0:
        print("can go next")
    elif X == -921:
        print("can go previous")
    else :
        print("can't continuee")

    cv2.waitKey(0)
    cv2.destroyAllWindows()



datetime = QDateTime.currentDateTime()

details = []

class Ui_window(object):
    global details

    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='university',
                                             user='root',
                                             password='')

        sql = "SELECT * FROM users WHERE current ='1'"
        cursor = connection.cursor()
        cursor.execute(sql)
        records = cursor.fetchall()

        print("Total number of rows is: ", cursor.rowcount)

        for row in records:
            details = row
            print (details)

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("connection is closed")


    def clickNext(self):
        print("Next notice!")



    def clickBack(self):
        print("Previous notice!")



    def setupUi(self, window):
        window.setObjectName("window")
        window.resize(950, 600)
        font = QtGui.QFont()
        window.setFont(font)
        window.setStyleSheet("background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:1, y2:1, stop:0 rgba(13, 0, 6, 255), stop:0.0298507 rgba(255, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));")
        self.lblUser = QtWidgets.QLabel(window)
        self.lblUser.setGeometry(QtCore.QRect(590, 10, 130, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        QtCore.QMetaObject.connectSlotsByName(window)

        window.setWindowFlag(Qt.WindowMinimizeButtonHint, False)

        window.setWindowFlag(Qt.WindowMaximizeButtonHint, False)

        window.setWindowFlag(Qt.WindowCloseButtonHint, False)

        

        self.lblUser.setFont(font)
        self.lblUser.setStyleSheet("color: rgb(255, 255, 255);\n")
        self.lblUser.setObjectName("lblUser")
        self.lbl_ShowUser = QtWidgets.QLabel(window)
        self.lbl_ShowUser.setGeometry(QtCore.QRect(725, 10, 200, 20))

        ####################################################

        self.lbl_ShowUser.setText(str(details[0]))

        ####################################################

        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbl_ShowUser.setFont(font)
        self.lbl_ShowUser.setStyleSheet("color: rgb(255, 255, 255);\n"
                                   "color: rgb(255, 255, 255);")
        self.lblOutOf = QtWidgets.QLabel(window)
        self.lblOutOf.setGeometry(QtCore.QRect(380, 10, 75, 20))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.lblOutOf.setFont(font)
        self.lblOutOf.setStyleSheet("color: rgb(0, 0, 255);\n"
"color: rgb(255, 255, 255);")
        self.lblOutOf.setObjectName("lblOutOf")
        self.lbl_AvailNo = QtWidgets.QLabel(window)
        self.lbl_AvailNo.setGeometry(QtCore.QRect(470, 6, 35, 25))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_AvailNo.setFont(font)
        self.lbl_AvailNo.setStyleSheet("color: rgb(0, 0, 255);\n"
"color: rgb(255, 255, 255);")
        self.lbl_AvailNo.setObjectName("lbl_AvailNo")
        self.lbl_CurrentNo = QtWidgets.QLabel(window)
        self.lbl_CurrentNo.setGeometry(QtCore.QRect(350, 6, 28, 25))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(100)
        self.lbl_CurrentNo.setFont(font)
        self.lbl_CurrentNo.setStyleSheet("color: rgb(0 , 0, 255);\n"
"color: rgb(255, 0, 0);")
        self.lbl_CurrentNo.setText(str(details[1]))
        self.lbl_CurrentNo.setObjectName("lbl_CurrentNo")
        y = (int(details[1])/20)*100
        self.lblDate = QtWidgets.QLabel(window)
        self.lblDate.setGeometry(QtCore.QRect(20, 10, 270, 20))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblDate.setFont(font)
        self.lblDate.setText(datetime.toString())
        self.lblDate.setObjectName("lblDate")
        self.lblDate.setStyleSheet("QLabel { background-color : gray; color : white; }");

        self.lblFaculty = QtWidgets.QLabel(window)
        self.lblFaculty.setGeometry(QtCore.QRect(20, 50, 100, 30))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.lblFaculty.setFont(font)
        self.lblFaculty.setObjectName("lblFaculty")
        self.lblFaculty.setStyleSheet("color: rgb(0 , 0, 255);\n"
                                         "color: rgb(255, 255, 255);")

        self.lblDep = QtWidgets.QLabel(window)
        self.lblDep.setGeometry(QtCore.QRect(20, 85, 130, 30))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.lblDep.setFont(font)
        self.lblDep.setObjectName("lblDep")
        self.lblDep.setStyleSheet("color: rgb(0 , 0, 255);\n"
                                         "color: rgb(255, 255, 255);")
        self.lbl_ShowFaculty = QtWidgets.QLabel(window)
        self.lbl_ShowFaculty.setGeometry(QtCore.QRect(190, 50, 221, 30))
        self.lbl_ShowFaculty.setText(str(details[2]))
        self.lbl_ShowFaculty.setObjectName("lbl_ShowFaculty")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbl_ShowFaculty.setFont(font)
        self.lbl_ShowFaculty.setStyleSheet("color: rgb(0 , 0, 255);\n"
                                         "color: rgb(255, 255, 255);")

        # self.lbl_ShowRegNo = QtWidgets.QLabel(window)
        # self.lbl_ShowRegNo.setGeometry(QtCore.QRect(190, 80, 221, 30))
        # self.lbl_ShowRegNo.setText("ICT/15/16/080")
        # self.lbl_ShowRegNo.setObjectName("lbl_ShowRegNo")
        # self.lbl_ShowRegNo.setStyleSheet("color: rgb(0 , 0, 255);\n"
        #                                  "color: rgb(255, 255, 255);")
        self.lbl_ShowDep = QtWidgets.QLabel(window)
        self.lbl_ShowDep.setGeometry(QtCore.QRect(190, 85, 400, 30))
        self.lbl_ShowDep.setText(str(details[3]))
        self.lbl_ShowDep.setObjectName("lbl_ShowDep")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbl_ShowDep.setFont(font)
        self.lbl_ShowDep.setStyleSheet("color: rgb(0 , 0, 255);\n"
                                         "color: rgb(255, 255, 255);")
        self.btnPrev = QtWidgets.QPushButton(window)
        self.btnPrev.setGeometry(QtCore.QRect(15, 320, 75, 70))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnPrev.setFont(font)
        self.btnPrev.setStyleSheet("color: rgb(255, 255, 255);")
        self.btnPrev.setObjectName("btnPrev")
        try{
            
        }
        self.btnPrev.clicked.connect(self.clickBack)

        self.btnNext = QtWidgets.QPushButton(window)
        self.btnNext.setGeometry(QtCore.QRect(870, 320, 75, 70))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnNext.setFont(font)
        self.btnNext.setStyleSheet("color: rgb(255, 255, 255);")
        self.btnNext.setObjectName("btnNext")
        self.btnNext.clicked.connect(self.clickNext)

        self.btnQuit = QtWidgets.QPushButton(window)
        self.btnQuit.setGeometry(QtCore.QRect(145, 535, 70, 30))
        self.btnQuit.setStyleSheet("color: rgb(0, 0, 0);")
        self.btnQuit.setObjectName("btnQuit")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btnQuit.setFont(font)
        self.btnQuit.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0.0153731, y1:0.018, x2:0, y2:0, stop:0 rgba(255, 7, 0, 255), stop:0.0298507 rgba(255, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));")
        self.btnQuit.clicked.connect(window.close)

        self.lblContent = QtWidgets.QLabel(window)
        self.lblContent.setGeometry(QtCore.QRect(95, 180, 770, 340))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lblContent.setText(str(details[5]))
        # self.lblContent.setText("There is a HCI Lecture for all ICT students tommorow.. \nVenue: L11 \nTime: 9.00-12.00a.m")
        self.lblContent.setObjectName("lblContent")
        self.lblContent.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lblContent.setFont(font)
        self.lblContent.setStyleSheet("QLabel { background-color : qlineargradient(spread:pad, x1:1, y1:1, x2:1, y2:1, stop:0 rgba(44, 43, 43, 237), stop:0.0298507 rgba(255, 0, 0, 255), stop:1 rgba(255, 255, 255, 255)); color : white; }");

        self.lbl_topic = QtWidgets.QLabel(window)
        self.lbl_topic.setGeometry(QtCore.QRect(95, 150, 770, 25))
        self.lbl_topic.setText(str(details[4]))
        self.lbl_topic.setObjectName("lbl_topic")
        self.lbl_topic.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(85)
        self.lbl_topic.setFont(font)
        self.lbl_topic.setStyleSheet("background-color: qlineargradient(spread:pad, x1:1, y1:1, x2:1, y2:1, stop:0 rgba(44, 43, 43, 237), stop:0.0298507 rgba(255, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));color : white; }")

        self.progressBar = QtWidgets.QProgressBar(window)
        self.progressBar.setGeometry(QtCore.QRect(375, 540, 118, 23))
        self.progressBar.setProperty("value", y)
        self.progressBar.setObjectName("progressBar")
        font = QtGui.QFont()
        font.setPointSize(12)
        self.progressBar.setFont(font)
        self.progressBar.setStyleSheet("color: rgb(0 , 0, 255);\n"
                                         "color: rgb(255, 255, 255);")
        self.lblExp = QtWidgets.QLabel(window)
        self.lblExp.setGeometry(QtCore.QRect(600, 540, 130, 25))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lblExp.setFont(font)
        self.lblExp.setStyleSheet("color: rgb(255, 255, 255);\n"
"color: rgb(255, 255, 255);")
        self.lblExp.setObjectName("lblExp")
        self.lbl_ShowExp = QtWidgets.QLabel(window)
        self.lbl_ShowExp.setGeometry(QtCore.QRect(740, 540, 150, 25))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_ShowExp.setFont(font)
        self.lbl_ShowExp.setStyleSheet("color: rgb(255, 255, 255);\n"
"color: rgb(255, 255, 255);")
        self.lbl_ShowExp.setText("2020/02/28")
        self.lbl_ShowExp.setObjectName("lbl_ShowExp")

        self.retranslateUi(window)


    def retranslateUi(self, window):
        _translate = QtCore.QCoreApplication.translate
        window.setWindowTitle(_translate("window", "Smart Notice Board"))
        self.lblUser.setText(_translate("window", "User Name :"))
        self.lblOutOf.setText(_translate("window", "Out Of"))
        self.lbl_AvailNo.setText(_translate("window", "20"))
        self.lblFaculty.setText(_translate("window", "Faculty :"))
        self.btnQuit.setText(_translate("window", "Exit"))
        self.lblDep.setText(_translate("window", "Department :"))
        self.btnPrev.setText(_translate("window", "<<"))
        self.btnNext.setText(_translate("window", ">>"))
        self.lblExp.setText(_translate("window", "Expire Date :"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QWidget()
    ui = Ui_window()
    ui.setupUi(window)
    # ex = Example()
    window.show()

    sys.exit(app.exec_())
