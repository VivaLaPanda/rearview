package main

import (
	"fmt"

	"gocv.io/x/gocv"
)

func main() {
	// set to use a video capture device 0
	deviceID := 0

	// open webcam
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer webcam.Close()

	// open display window
	window := gocv.NewWindow("Face Detect")
	defer window.Close()

	// prepare image matrix
	img := gocv.NewMat()
	defer img.Close()

	// load frontalClassifier to recognize faces
	frontalClassifier := gocv.NewCascadeClassifier()
	defer frontalClassifier.Close()
	profileClassifier := gocv.NewCascadeClassifier()
	defer profileClassifier.Close()

	if !frontalClassifier.Load("data/haarcascade_frontalface_default.xml") {
		fmt.Println("Error reading cascade file: data/haarcascade_frontalface_default.xml")
		return
	}

	if !profileClassifier.Load("data/lbpcascade_profileface.xml") {
		fmt.Println("Error reading cascade file: data/lbpcascade_profileface.xml")
		return
	}

	fmt.Printf("start reading camera device: %v\n", deviceID)
	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("cannot read device %v\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		// detect faces
		numfaces := len(frontalClassifier.DetectMultiScale(img))
		numfaces += len(profileClassifier.DetectMultiScale(img))
		fmt.Printf("found %d faces\n", numfaces)

		// show the image in the window, and wait 1 millisecond
		//window.IMShow(img)
		window.WaitKey(1)
	}
}
