#include "opencv_lib.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv/cv.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>
#include <iterator>

using namespace cv;
using namespace std;

#define SPACE_INTERRUPT true
#define FACE_WIDTH 92
#define FACE_HEIGHT 112
#define RESULT_WIDTH 512
#define RESULT_HEIGHT 512
#define RATIO_WIDTH 1.1f;
#define RATIO_HEIGHT 1.5f;




//帮助函数
static void help()
{
    cout << "Wrong command!\nFormat: detect.exe (example.jpg | example.avi)"<< endl;
	getchar();
}

//部分全局变量

//检测相关
string cascadeName = "haarcascade_frontalface_alt.xml";
string nestedCascadeName = "haarcascade_eye_tree_eyeglasses.xml";
CvMemStorage* storage = NULL;
CvMemStorage* my_cascadeMem=NULL;
CvHaarClassifierCascade* my_cascade = NULL;
CascadeClassifier cascade;
CascadeClassifier nestedCascade;
bool tryflip = false;
double scale = 1;


//匹配相关
Ptr<FaceRecognizer> model;
// These vectors hold the images and corresponding labels.
vector<Mat> images;
vector<int> labels;
vector<string> paths;
vector<string> names;

//int testLabel;
//Mat testSample;

//调整大小和转换为灰度值,传出结果为Mat
static Mat norm_0_255(IplImage* temp_img,int width,int height)
{
	Mat dst;
	//先调整大小
	IplImage* temp_resize_img = cvCreateImage(cvSize(width,height),temp_img->depth,temp_img->nChannels);
	cvResize(temp_img,temp_resize_img);
	if(temp_img->nChannels!=1)
	{
		//再转换为灰度图
		IplImage* temp_resize_gray_img = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
		cvCvtColor(temp_resize_img,temp_resize_gray_img, CV_BGR2GRAY);//灰度图像
		dst = Mat(temp_resize_gray_img);
		cvReleaseImage( &temp_resize_img );
	}
	else
	{
		dst = Mat(temp_resize_img);
	}
	cvReleaseImage( &temp_img );
	return dst;
}

//从图像orgImage中提取一块（rectInImage）子图像imgRect
IplImage* GetImageRect(IplImage* orgImage, CvRect rectInImage)
{
	IplImage *result = NULL;
	
	//从图像中提取子图像
	if(rectInImage.x<0)
	{
		rectInImage.x = 0;
	}
	if(rectInImage.y<0)
	{
		rectInImage.y = 0;
	}
	if(rectInImage.width + rectInImage.x > orgImage->width)
	{
		rectInImage.width = orgImage->width - rectInImage.x;
	}
	if(rectInImage.height + rectInImage.y > orgImage->height)
	{
		rectInImage.height = orgImage->height - rectInImage.y;
	}
	result=cvCreateImage( cvSize(rectInImage.width,rectInImage.height),
		orgImage->depth, orgImage->nChannels );//new IplImages*   用于存储一张检测出的人脸
	/*printf("orgImage(w = %d,h = %d)\n",orgImage->width,orgImage->height);
	printf("rectInImage(x = %d,y = %d,w = %d,h = %d)\n",rectInImage.x,rectInImage.y,rectInImage.width,rectInImage.height);
	printf("result(w = %d,h = %d)\n",result->width,result->height);*/
	cvSetImageROI(orgImage,rectInImage);
	cvCopy(orgImage,result);
	cvResetImageROI(orgImage);
	return result;
}

// 检测当前帧的人脸并得到人脸rect序列
CvSeq* detect( IplImage* img, CvSize minSize)
{
	// Create a new image based on the input image
	CvSeq* tempFaces;
	// Clear the memory storage which was used before
	cvClearMemStorage( storage );
	// Find whether the cascade is loaded, to find the faces. If yes, then:
	if( my_cascade )
	{
		// There can be more than one face in an image. So create a growable sequence of faces.
		// Detect the objects and store them in the sequence
		tempFaces = cvHaarDetectObjects( img, my_cascade, storage,1.1, 2, CV_HAAR_DO_CANNY_PRUNING,minSize );
	}
	return tempFaces;
}

// 检测当前帧的人脸并得到人脸rect序列同时绘制线框
vector<CvRect> better_detect( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{

	scale = 1.0f;
    int i = 0;
    double t = 0;
    vector<CvRect> result_faces;
	vector<Rect> faces,faces2;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CV_HAAR_FIND_BIGGEST_OBJECT
                                 //|CV_HAAR_DO_ROUGH_SEARCH
                                 |CV_HAAR_SCALE_IMAGE
                                 ,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
        {
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
            
        }
    }

	//这里可以投机取巧,若待检测图片尺寸小于库中脸的尺寸,则直接返回整张图片大小的矩形
	if(img.cols <= FACE_WIDTH && img.rows <= FACE_HEIGHT)
	{
		result_faces.push_back(cvRect(0, 0, img.cols, img.rows));
		return result_faces;
	}
	
	//将检测出的人脸转换为CvRect(适当地进行扩大)
	int new_width,new_height,new_x,new_y,center_x,center_y;
	for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++ )
    {
		center_x = r->x + r->width/2;
		center_y = r->y + r->height/2;
		new_width = r->width *RATIO_WIDTH;
		new_height = r->height *RATIO_HEIGHT;
		new_x = center_x - new_width/2;
		new_y = center_y - new_height/2 - new_height * 0.05f;
		if(new_x < 0)
		{
			new_x = 0;
		}
		if(new_y<0)
		{
			new_y = 0;
		}
		if(new_width + new_x > img.cols)
		{
			new_width = img.cols - new_x;
		}
		if(new_height + new_y > img.rows)
		{
			new_height = img.rows - new_y;
		}
		result_faces.push_back(cvRect(new_x, new_y, new_width, new_height));
    }

    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    return result_faces;
}



//输入:当前帧图像以及检测的出的人脸rect序列     作用:匹配并进行绘制当前帧的每一张脸和对应的数据库中的脸
void match_and_draw(IplImage* curFrame, vector<CvRect> faces)
{
	// Create two points to represent the face locations
	CvPoint pt1, pt2;
	int scale = 1;
	float ratio_x = 1.0f;
	float ratio_y = 1.0f;
	ratio_x = RESULT_WIDTH * 1.0f / curFrame->width;
	ratio_y = RESULT_HEIGHT * 1.0f / curFrame->height;
	//将帧画面统一输出为512*512的图像中
	IplImage* temp_res = NULL;
	temp_res = cvCreateImage( cvSize(RESULT_WIDTH,RESULT_HEIGHT), curFrame->depth, curFrame->nChannels );
	cvResize(curFrame,temp_res);
	//IplImage* temp_res = cvCreateImage( cvGetSize(curFrame), curFrame->depth, curFrame->nChannels );//new IplImages*  用于存储每一次检测的结果图并绘制出来
	
	//cvCopy(curFrame,temp_res,NULL);//将原始图像保存到temp_res中
	//CvRect* temp_rect;
	//cout<<"faces.size()!!!"<<endl;
	if(faces.size()==0)
	{
		cout<<"SOORY,没有检测到人脸,换个角度试试!"<<endl;
		/*printf("curFrame(w = %d,h = %d,depth = %d,channel = %d)\n",
					curFrame->width,curFrame->height,curFrame->depth,curFrame->nChannels);*/
	}
	else
	{
		cout<<"检测到人脸数:"<<faces.size()<<endl;
	}
	// Loop the number of faces found.
	
	CvFont font;double hScale=0.5;double vScale=1.0;int lineWidth=2;
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC,hScale,vScale,0,lineWidth);
	for( unsigned int i = 0; i < faces.size(); i++ )
	{
		//从图像中把检测窗口转为图像
		IplImage* detectFace;
		detectFace = GetImageRect(curFrame, faces[i]);//得到一个检测出的人脸数据
		
		/*if(i==0)
		{
			cvShowImage( "test0", detectFace );
		}
		else if(i==1)
		{
			cvShowImage( "test1", detectFace );
		}
		else
		{
			cvShowImage( "test2", detectFace );
		}*/

		// 在temp_res上绘制对应的矩形框
		pt1.x = int(faces[i].x*ratio_x);
		pt2.x = int((faces[i].x + faces[i].width)*ratio_x);
		pt1.y = int(faces[i].y*ratio_y);
		pt2.y = int((faces[i].y + faces[i].height)*ratio_y);
		cvRectangle( temp_res, pt1, pt2, CV_RGB(255,0,0), 2, 8, 0 );
		
		//匹配人脸并返回结果
		int predictedLabel = -1;
		double confidence = 0.0;
		//model->set("threshold", 0.0);
		model->predict(norm_0_255(detectFace,FACE_WIDTH ,FACE_HEIGHT ), predictedLabel, confidence);//将人头像都变为92*112在进行匹配
		if(predictedLabel==-1)
		{
			cout<<"匹配失败!"<<endl;
		}
		else
		{
			for(unsigned int j = 0;j < labels.size();++j)
			{
				if(labels[j] == predictedLabel)//简单起见,先找到的直接绘制
				{
					//得到数据库中头像
					IplImage *temp_photo = cvLoadImage(paths[j].c_str());
					//将头像调整大小为(FACE_WIDTH,FACE_HEIGHT)
					IplImage* temp_resize_img = cvCreateImage(cvSize(FACE_WIDTH,FACE_HEIGHT),temp_photo->depth,temp_photo->nChannels);
					cvResize(temp_photo,temp_resize_img);
					//printf("temp_resize_img(w = %d,h = %d,depth = %d,channel = %d)\n",
					//	temp_resize_img->width,temp_resize_img->height,temp_resize_img->depth,temp_resize_img->nChannels);
					//
					////cout<<"1 cvCopy(temp_resize_img,CurFrame);"<<endl;
					//printf("curFrame(w = %d,h = %d,depth = %d,channel = %d)\n",
					//	curFrame->width,curFrame->height,curFrame->depth,curFrame->nChannels);
					//将头像嵌入到当前帧画面
					int temp_x = int(faces[i].x * ratio_x) - FACE_WIDTH;
					if(temp_x<0)
					{
						temp_x = 0;
					}
					int temp_y = int(faces[i].y * ratio_y);
					{
						if(temp_y + FACE_HEIGHT > RESULT_HEIGHT)
						{
							temp_y = RESULT_HEIGHT - FACE_HEIGHT;
						}
					}
				
					//printf("temp_resize_img(w = %d,h = %d,depth = %d,channel = %d)\n",
					//	temp_resize_img->width,temp_resize_img->height,temp_resize_img->depth,temp_resize_img->nChannels);
				
					cvSetImageROI(temp_res,cvRect(temp_x,temp_y,FACE_WIDTH,FACE_HEIGHT));
				
					//cout<<"temp_x;"<<temp_x<<endl;
					//cout<<"temp_y;"<<temp_y<<endl;

					cvCopy(temp_resize_img,temp_res);
					//cout<<"2 cvCopy(temp_resize_img,CurFrame);"<<endl;
					cvResetImageROI(temp_res);
					//绘制姓名
					cout<<" 检测结果: 人物编号:"<<predictedLabel<<"姓名:"<<names[j]<<endl;
					cvPutText(temp_res,names[j].c_str(),cvPoint(int(faces[i].x * ratio_x),int(faces[i].y*ratio_y)-10),
						&font,cvScalar(255,0,0));
					//cvDrawText();names[j];
					break;
				}
			}
		}

		
		//cvWaitKey(0);
		//cvReleaseImage( &temp_res);
		//cvReleaseImage( &detectFace );	
	}
	cvShowImage( "result", temp_res );
	return;
}


void DetectAndDraw(IplImage* curFrame , CvSize minSize)
{
	//curFrame:待检测的图像；winSize：检测窗口大小
	vector<CvRect> faces;//储存人脸rect序列
	// Check whether the cascade has loaded successfully. Else report and error and quit
	if( curFrame!=NULL )
	{
		cout<<"人脸检测开始!"<<endl;
		faces = better_detect( Mat(curFrame), cascade, nestedCascade, scale,tryflip );
		cout<<"人脸检测结束!"<<endl;
		match_and_draw(curFrame, faces);
		cout<<"人脸匹配结束!"<<endl;
	}
	//cvRelease((void**)&faces);
	//cvReleaseImage( curFrame );
}


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';')
{
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file)
	{
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path,classlabel,name;
	//paths
    while (getline(file, line))
	{
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel,separator);
		getline(liness, name);

		//IplImage* temp_img = NULL;
		if(!path.empty() && !classlabel.empty())
		{
			paths.push_back(path);
			names.push_back(name);
			IplImage *temp_img = cvLoadImage(path.c_str());
			if(temp_img)
			{
				images.push_back(imread(path, 0));
				//images.push_back(norm_0_255(temp_img,FACE_WIDTH ,FACE_HEIGHT ));//将人头像都变为128*128的灰度图
				labels.push_back(atoi(classlabel.c_str()));
			}
        }
    }
}

//建立人脸匹配模型的函数
void initMatchMode(string fn_csv)
{
	try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
		cout<< "Error occurs when opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		cout<<"Press any key to exit!"<<endl;
		getchar();
		exit(1);
        //cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
    }

	if(images.size() <= 1)
	{
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }

	
	cout<<"人脸匹配模型训练中......"<<endl;
	//建立检测模型并进行训练
	//model = createEigenFaceRecognizer();
	//model->train(images, labels);
	model = createFisherFaceRecognizer();
	model->train(images, labels);
	//model = createLBPHFaceRecognizer();
	//model->train(images, labels);
	
	cout<<"人脸匹配模型训练结束!"<<endl;
}

//建立人脸检测模型的函数
void initDetectMode()
{
	// Load the HaarClassifierCascad
	my_cascade = (CvHaarClassifierCascade*)cvLoad( cascadeName.c_str(), my_cascadeMem, 0, 0 );
	// Check whether the cascade has loaded successfully. Else report and error and quit
	if( !my_cascade )
	{
		cout<<"Could not load classifier cascade/n"<<endl;
		getchar();
		return;
	}
	// Allocate the memory storage
	storage = cvCreateMemStorage(0);
}

//目标:读一张图像,检测出人脸,在数据库中搜索出对应的人,在图片上显示人的名字并在左侧显示一张最相近的小图
int main(int argc, const char *argv[])
{
	////*******************得到IplImage* CurFrame *******************************detection
    const string scaleOpt = "--scale=";
    size_t scaleOptLen = scaleOpt.length();
    const string cascadeOpt = "--cascade=";
    size_t cascadeOptLen = cascadeOpt.length();
    const string nestedCascadeOpt = "--nested-cascade";
    size_t nestedCascadeOptLen = nestedCascadeOpt.length();
    const string tryFlipOpt = "--try-flip";
    size_t tryFlipOptLen = tryFlipOpt.length();
    string inputName;
	
	CvCapture* capture = 0;
	IplImage* CurFrame = NULL;
	int detect_width = 100;
	int detect_height = 100;


    for( int i = 1; i < argc; i++ )
    {
        cout << "Processing " << i << " " <<  argv[i] << endl;
        if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 )
        {
            cascadeName.assign( argv[i] + cascadeOptLen );
            cout << "  from which we have cascadeName= " << cascadeName << endl;
        }
        else if( nestedCascadeOpt.compare( 0, nestedCascadeOptLen, argv[i], nestedCascadeOptLen ) == 0 )
        {
            if( argv[i][nestedCascadeOpt.length()] == '=' )
                nestedCascadeName.assign( argv[i] + nestedCascadeOpt.length() + 1 );
            if( !nestedCascade.load( nestedCascadeName ) )
                cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
        }
        else if( scaleOpt.compare( 0, scaleOptLen, argv[i], scaleOptLen ) == 0 )
        {
            if( !sscanf( argv[i] + scaleOpt.length(), "%lf", &scale ) || scale < 1 )
                scale = 1;
            cout << " from which we read scale = " << scale << endl;
        }
        else if( tryFlipOpt.compare( 0, tryFlipOptLen, argv[i], tryFlipOptLen ) == 0 )
        {
            tryflip = true;
            cout << " will try to flip image horizontally to detect assymetric objects\n";
        }
        else if( argv[i][0] == '-' )
        {
            cerr << "WARNING: Unknown option %s" << argv[i] << endl;
        }
        else
            inputName.assign( argv[i] );
    }

	//输入判断结束
	//**************初始化检测模型****************
	//initDetectMode();	
	//**************初始化匹配模型****************
	
	initMatchMode("at.txt");
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
		getchar();
        return -1;
    }
    if( inputName.empty() || (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0') )
    {
        capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' );
        int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;
        if(!capture)
			cout << "Capture from CAM " <<  c << " didn't work" << endl;
    }
    else if( inputName.size() )
    {
        //image = imread( inputName, 1 );
		CurFrame = cvLoadImage( inputName.c_str());
        if( CurFrame==NULL )
        {
            capture = cvCaptureFromAVI( inputName.c_str() );
            if(!capture) cout << "Capture from AVI didn't work" << endl;
        }
    }
    else
    {
        //image = imread( "lena.jpg", 1 );
		CurFrame = cvLoadImage("a1.jpg");
        if( CurFrame==NULL ) cout << "Couldn't read a1.jpg" << endl;
    }

	//***************************************************开始检测**********************************************
    
	cvNamedWindow( "result", 1 );
	/*cvNamedWindow( "test0", 1 );
	cvNamedWindow( "test1", 1 );
	cvNamedWindow( "test2", 1 );*/
    if( capture )
    {//视频或者摄像头提取图像过程中
        cout << "视频读取中..." << endl;
        for(;;)
        {
            CurFrame = cvQueryFrame( capture );
			//CurFrame = cvQueryFrame( capture );//有的机器可能需要读取两次才能将摄像中的图片读出来
            if( CurFrame == NULL)
                break;
			DetectAndDraw(CurFrame , cvSize(detect_width,detect_height));//检测人脸并绘制出对应的帧,红框以及小头像
			
			int key_input = waitKey(10);
			if( key_input == 27)
			{
				cout<<"esc is pressed!"<<endl;
				goto _cleanup_;
			}
			else
			{
				cout<<"key_input = "<<key_input<<endl;
				if(SPACE_INTERRUPT)
				{
					if( key_input !=32)
					{
						while((key_input = waitKey(10)) != 32)
						{
							if(key_input == 27)
							{
								goto _cleanup_;
							}
						}
					}
				}
			}
			//detectAndDraw( Mat(CurFrame), cascade, nestedCascade, scale, tryflip );
		}
_cleanup_:
        cvReleaseCapture( &capture );
    }
    else
    {
        if( CurFrame!=NULL )
        {
			//从图像文件检测
			cout << "图片读取中..." << endl;
			DetectAndDraw(CurFrame , cvSize(detect_width,detect_height));
            //detectAndDraw( Mat(CurFrame), cascade, nestedCascade, scale, tryflip );//检测人脸并绘制出对应的帧,红框以及小头像s
        }
        else if( !inputName.empty() )
        {
			////读取不正确格式文件的操作,,可以不用管**********
			cout<<"文件格式不支持!"<<endl;
			exit(1);
        }
		else
		{}
    }
	//***************************************************检测结束**********************************************
	cout<<"按任意键退出!"<<endl;
    waitKey(0);
	cvDestroyWindow("result");
	/*cvDestroyWindow( "test0");
	cvDestroyWindow( "test1");
	cvDestroyWindow( "test2");*/
	return 0;
}