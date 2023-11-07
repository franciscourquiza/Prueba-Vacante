using System;
using System.IO;
using Emgu.CV.CvEnum;
using OpenCvSharp;

class Program
{
    static void Main(string[] args)
    {
        string videoPath1 = "C:/Users/cvpru/OneDrive/Escritorio/PruebaParaVacante/PruebaParaVacante/video.mp4";  // Primer video
        string videoPath2 = "C:/Users/cvpru/OneDrive/Escritorio/PruebaParaVacante/PruebaParaVacante/video2.mp4";  // Segundo video
        string outputFolder = "C:/Users/cvpru/OneDrive/Escritorio/PruebaParaVacante/PruebaParaVacante/differences";  // Carpeta de salida

        CascadeClassifier faceCascade = new CascadeClassifier("C:/Users/cvpru/OneDrive/Escritorio/PruebaParaVacante/PruebaParaVacante/haarcascade_frontalface_default.xml");  //ruta de acceso del haarscade xml

        using (VideoCapture capture1 = new VideoCapture(videoPath1))
        using (VideoCapture capture2 = new VideoCapture(videoPath2))
        {
            if (!capture1.IsOpened() || !capture2.IsOpened())
            {
                Console.WriteLine("No se pudo abrir uno de los videos.");
                return;
            }

            int fps1 = (int)capture1.Get(VideoCaptureProperties.Fps);
            int fps2 = (int)capture2.Get(VideoCaptureProperties.Fps);
            int frameCount = Math.Min(fps1 * 10, fps2 * 10);  // Elige el menor de los dos conteos de cuadros

            for (int i = 0; i < frameCount; i++)
            {
                using (Mat frame1 = new Mat())
                using (Mat frame2 = new Mat())
                {
                    if (!capture1.Read(frame1) || !capture2.Read(frame2))
                        break;

                    using (Mat grayFrame1 = new Mat())
                    using (Mat grayFrame2 = new Mat())
                    {
                        Cv2.CvtColor(frame1, grayFrame1, ColorConversionCodes.BGR2GRAY);
                        Cv2.CvtColor(frame2, grayFrame2, ColorConversionCodes.BGR2GRAY);

                        Rect[] faces1 = faceCascade.DetectMultiScale(grayFrame1, 1.1, 3, (HaarDetectionTypes)HaarDetectionType.ScaleImage, new Size(30, 30));
                        Rect[] faces2 = faceCascade.DetectMultiScale(grayFrame2, 1.1, 3, (HaarDetectionTypes)HaarDetectionType.ScaleImage, new Size(30, 30));

                        // Compara las caras detectadas
                        for (int j = 0; j < Math.Min(faces1.Length, faces2.Length); j++)
                        {
                            // Asegúrate de que las matrices de caras tengan el mismo tamaño
                            Mat face1 = grayFrame1[faces1[j]];
                            Mat face2 = grayFrame2[faces2[j]];

                            if (face1.Size() == face2.Size())
                            {
                                // Encuentra la diferencia entre las caras
                                Mat difference = new Mat();
                                Cv2.Absdiff(face1, face2, difference);

                                string framePath = Path.Combine(outputFolder, $"difference_{i:D5}_{j:D2}.jpg");
                                difference.SaveImage(framePath);
                            }
                        }
                    }
                }
            }
        }

        Console.WriteLine("Detección de caras y generación de imágenes de diferencias completadas.");
    }
}
