import cv2, os

'''
#use timeF
save_path=r"D:\CowRestAPI\test"
path = r"D:\CowRestAPI\test"
filelist = os.listdir(path)

for item in filelist:
    if item.endswith('.mp4'):
        print(item)
        video_to_picture_path= os.path.join(save_path, item.split(".")[0])
        try:
            if not os.path.exists(video_to_picture_path):
                os.makedirs(video_to_picture_path)
                src = os.path.join(path, item)
				# 入视频文件
                vc = cv2.VideoCapture(src)
			    c = 1

			if vc.isOpened():        #判断是否正常打开
				rval , frame = vc.read()
			else:
				rval = False

			timeF = 6 #视频帧计数间隔频率
			m = 1
			while rval:  #循环读取视频帧
				rval, frame = vc.read()
				# print(frame)
				(h, w) = frame.shape[:2]
				center = (w / 2, h / 2)
				# print(center)
				M = cv2.getRotationMatrix2D(center, -90, 1.0)
				rotated = cv2.warpAffine(frame, M, (w, h))

				if(c%timeF == 0):  #每隔timeF帧进行存储操作
					cv2.imwrite(video_to_picture_path + "/" + str(m) + '.jpg', rotated)#存储为图像
					m = m + 1
				c = c + 1
				cv2.waitKey(1)
			vc.release()
		except:
			print("error")
'''
