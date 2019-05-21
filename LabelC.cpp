#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include<io.h>
#include<cmath>

using namespace std;
using namespace cv;

struct UserData
{
	cv::Mat src;
	vector<cv::Point> pts;
};

static Point first_pt = Point(0, 0);
Mat mask_helper;
Mat mask_helper_save;
Point ptStart;
Mat temp;
list<string> file_paths;
string img_name;
bool Left_double_click = false;
UserData* d;

void on_mouse(int event, int x, int y, int flags, void* dp)
{
	Left_double_click = false;

	d = (UserData*)dp;

    temp = mask_helper.clone();

	if (event == CV_EVENT_LBUTTONDBLCLK) {  //左键双击， 完成当前轮廓绘制

		if (d->pts.size() > 2) {
			Left_double_click = true;
			d->pts.push_back(first_pt);
			first_pt = Point(0, 0);
		}
	}

	else if (event == CV_EVENT_RBUTTONDOWN)  //右键单击 取消上一个点选
	{		
		if (d->pts.size() > 0) {
			d->pts.pop_back();
		}
		if (d->pts.size() < 1) {
			first_pt = Point(0, 0);
			Left_double_click = false;		
		}
	}
	else if (event == CV_EVENT_LBUTTONDOWN)	{  //左键单击 点选

		if (first_pt.x == 0 && first_pt.y == 0) {
			first_pt = Point(x, y);
		}
		d->pts.push_back(Point(x, y));
	}		

	Point pp = Point(0, 0);

	for (int i = 0; i < (d->pts.size()); i++)
	{
		if (Left_double_click) {
			line(mask_helper, d->pts[i], d->pts[min(i + 1, int(d->pts.size() - 1))], Scalar(255, 255, 255), 1, CV_AA);
			line(mask_helper_save, d->pts[i], d->pts[min(i + 1, int(d->pts.size() - 1))], Scalar(255), 1, CV_AA);
		}
		pp = d->pts[min(i + 1, int(d->pts.size() - 1))];
		if (i + 1 > int(d->pts.size()-1)) {
			pp = Point(x, y);
		}
		line(temp, d->pts[i], pp, Scalar(255, 255, 255), 1, CV_AA);
	}
	
	circle(temp, Point(x, y), 1, Scalar(0, 255, 0), 1, 16); //鼠标当前绿点显示
	imshow("Label Contours", temp);

	if (Left_double_click && d->pts.size() > 2) {
		d->pts.clear();
	}
}

int label_img(Mat& src, Mat& mask, Mat &mask_helper, string& name)
{
	char c = 'x';
	img_name = name;
	vector<vector<Point> > poly_point_set;
	UserData d_copy;

	mask_helper = src.clone();

	while (c != 's')
	{
		UserData d;
		d.src = src.clone();

		namedWindow("Label Contours", 0);

		setMouseCallback("Label Contours", on_mouse, &d);

		if (mask_helper.data) {
			imshow("Label Contours", mask_helper);
		
		}
		else {
			imshow("Label Contours", src);		
		}		
		d_copy = d;
		c = waitKey(0);
		
	}

	if (d_copy.pts.size() > 0)
	{
		const cv::Point* ppt[1] = { &d_copy.pts[0] };
		int npt[] = { static_cast<int>(d_copy.pts.size()) };
		//cv::fillPoly(src, ppt, npt, 1, cv::Scalar(0, 0, 255), 16);
		cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255), 16);
		cv::fillPoly(mask_helper_save, ppt, npt, 1, cv::Scalar(255), 16);
		poly_point_set.push_back(d_copy.pts);
	}
	return 0;
}

/***************************************************
功能：以ptStart为起点对图像进行填充
参数：src-边界图像
	  ptStart-种子点的坐标
****************************************************/
void BoundarySeedFill(Mat& src, Point ptStart)
{
	Mat dst = Mat::zeros(src.size(), src.type());
	//int se[3][3] = { { -1, 1, -1 }, { 1, 1, 1 }, { -1, 1, -1 } };//十字形结构元素
	int se[3][3] = { { -1, -1, -1 }, { -1, 1, -1 }, { -1, -1, -1 } };//十字形结构元素

	Mat M_se = (Mat_<double>(3, 3) << -1, 1, -1, 1, 1, 1, -1, 1, -1);

	Mat tempImg = Mat::ones(src.size(), src.type()) * 255;
	Mat revImg = tempImg - src;//原图像的补集
	dst.at<uchar>(ptStart.y, ptStart.x) = 500;//绘制种子点
	Mat dif = Mat::zeros(src.size(), src.type());
	Mat temp;

	Mat dif2 = Mat::zeros(src.size(), src.type());

	int interval = 0;

	while (true)//循环膨胀图像直到图像不在产生变化
	{
		dst.copyTo(temp);
		Mat structElement1 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
		//dilate(dst, dst, structElement1); //用十字结构元素膨胀
		dilate(dst, dst, M_se); //用十字结构元素膨胀

		dst = dst & revImg; //限制膨胀不会超过原始边界
		subtract(dst, temp, dif);

		// 判断结果不再变化表示绘制完成

		if (interval % 10 == 0) {
			if (memcmp(dif2.data, dif.data, dif.total() * dif.elemSize()) == 0) {
				break;			}
		}
		//cout << interval << endl;
		interval++;
	}
	src = temp;
}

void fill_contours(cv::Mat& contorus_img) {	
	Point p;
	p.x = 0;
	p.y = 0;

	BoundarySeedFill(contorus_img, p);
	contorus_img = ~contorus_img;
}

void read_images(string filePath="D:\\test_img",string fileType = "\\*.*") {
	__int64  Handle;	
	struct __finddata64_t  FileInfo;
	//filename += fileType;
	if ((Handle = _findfirst64((filePath +fileType).data(), &FileInfo)) == -1L)
		cout << "没有找到匹配的项目" << endl;
	else
	{
		//cout<< FileInfo.name;
		while (_findnext64(Handle, &FileInfo) == 0) {
			string fileName = FileInfo.name;
			file_paths.push_back(fileName);
		}
		_findclose(Handle);
	}
}

int main(int argc, char* argv[])
{
	string root_path = argv[1];
	read_images(root_path);

	string command;
	string save_folderPath;
	save_folderPath =argv[2];

	const char* path_exist = save_folderPath.data();
	if (_access(path_exist, 0)) {
	    command = "mkdir -p " + save_folderPath;		
	    system(command.c_str());		
	}

	for (list<string>::iterator iter = file_paths.begin(); iter != file_paths.end(); iter++) {

		try {
			first_pt = Point(0, 0);
			string img_name = *iter;			


			cv::Mat src = cv::imread(root_path + "\\" + img_name);

			if (src.data == NULL) {
				continue;
			}

			mask_helper = Mat::zeros(src.size(), CV_8UC1);
			mask_helper_save = Mat::zeros(src.size(), CV_8UC1);

			cv::Mat mask(src.rows, src.cols, CV_8UC1);

			mask.setTo(0);

			label_img(src, mask, mask_helper, img_name);// label based on tiny

			Point p;

			p.x = 1;
			p.y = 1;

			BoundarySeedFill(mask_helper_save, p);
			mask_helper_save = ~mask_helper_save;

			threshold(mask_helper_save, mask_helper_save, 20, 255, CV_THRESH_BINARY);

			string save_path = save_folderPath + "\\" + img_name;
			imwrite(save_path, mask_helper_save);
		}
		catch (char* str) {
			cout << str;
		}
		cout << *iter << endl;	
	}
	return 0;
}