import cv2
import numpy as np
import os
import DetectChars
import DetectPlates
import PossiblePlate
import sqlite3

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
showSteps = False


def main():
    count = 0
    a = 0
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):  # True
        ret, imgOriginalScene = cap.read()  # open image
        cv2.imshow('Camera', imgOriginalScene)
        count = count + 1
        if count == 10:
            count = 0
            blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()  # attempt KNN training
            listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)  # detect plates
            listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)  # detect chars in plates
            cv2.imshow('Camera', imgOriginalScene)
            # cv2.imshow("imgOriginalScene", imgOriginalScene)  # show scene image
            if len(listOfPossiblePlates) == 0:  # if no plates were found
                print("\nno license plates were detected\n")  # inform user no plates were found
            else:
                listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
                licPlate = listOfPossiblePlates[0]
                cv2.imshow('Camera', imgOriginalScene)
                data = str(licPlate.strChars)
                try:
                    data = int(data)
                    data = str(data)
                    i = 0
                    for a in data:
                        i = i + 1
                    if i >= 4:
                        data = data[0:4]
                        connection = sqlite3.connect("License.db")
                        con = connection.cursor()
                        sql_cmd = "SELECT * FROM NoPlates where reg = " + str(data)
                        con.execute(sql_cmd)
                        result = con.fetchall()
                        if not result:
                            print("Not Found in Database")

                            cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
                            cv2.imshow("imgThresh", licPlate.imgThresh)
                            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)  # draw red rectangle around plate
                            # print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
                            writeLicensePlateCharsOnImage(imgOriginalScene,licPlate)  # write license plate text on the image
                            print("----------------------------------------")
                            cv2.imwrite("NotFoundPlatesImg\img" + str(data) + ".png", imgOriginalScene)  # write image out to file
                            cv2.imshow('Camera', imgOriginalScene)

                        else:
                            for r in result:
                                format_str = "License No# {first} Name {second} Contact {third} Designation {forth}"
                                s = format_str.format(first=r[0], second=r[1], third=r[2], forth=r[3])
                                print(s)
                            cv2.imshow("imgPlate", licPlate.imgPlate)  # show crop of plate and threshold of plate
                            cv2.imshow("imgThresh", licPlate.imgThresh)
                            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)  # draw red rectangle around plate
                            # print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # write license plate text to std out
                            writeLicensePlateCharsOnImage(imgOriginalScene,
                                                          licPlate)  # write license plate text on the image
                            print("----------------------------------------")
                            cv2.imwrite("FoundPlatesImg\img" + str(data) + ".png", imgOriginalScene)  # write image out to file
                            cv2.imshow('Camera', imgOriginalScene)
                except:
                    pass
            cv2.imshow('Camera', imgOriginalScene)
            # cv2.waitKey(0)  # hold windows open until user presses a key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)  # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0  # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0  # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX  # choose a plain jane font
    fltFontScale = float(plateHeight) / 30.0  # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))  # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                         intFontThickness)  # call getTextSize

    # unpack roatated rect into center point, width and height, and angle
    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)  # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)  # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
            round(plateHeight * 1.6))  # write the chars in below the plate
    else:  # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
            round(plateHeight * 1.6))  # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize  # unpack text size width and height

    ptLowerLeftTextOriginX = int(
        ptCenterOfTextAreaX - (textSizeWidth / 2))  # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(
        ptCenterOfTextAreaY + (textSizeHeight / 2))  # based on the text area center, width, and height

    # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_YELLOW, intFontThickness)


if __name__ == "__main__":
    main()
