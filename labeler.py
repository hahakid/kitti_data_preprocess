#import pykitti
import os
import cv2
import numpy as np
import utils

filedir='../'
model=['object','tracking']
obj_train_dir=['calib','image_2','label_2','velodyne']
obj_test_dir=['calib','image_2','velodyne']

#tracking_seq='0000' #train: 0000-0020, test: 0000-0028
show_flag=True # visualize
tracking_train_seq=20 #20 #20
tracking_test_seq=28
#original
original_class_dic = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6}
#combined
combine_class_dic = {'Car': 0, 'Van': 0, 'Truck': 0, 'Pedestrian': 1, 'Person_sitting': 1, 'Cyclist': 2, 'Tram': 0}

def tracking_data(path,mode='training'):
    for i in range(0,tracking_train_seq+1):
        seq_id=str(i).zfill(4)
        if mode=='training':
            if not os.path.exists(os.path.join(path, mode,'front_view_label')):
                os.mkdir(os.path.join(path, mode,'front_view_label'))

            save_path=os.path.join(path, mode,'front_view_label',seq_id)
            path_image=os.path.join(path, mode,'image_02',seq_id)
            path_velo=os.path.join(path, mode,'velodyne',seq_id)
            path_oxts=os.path.join(path, mode,'oxts',seq_id+'.txt')
            path_label=os.path.join(path, mode,'label_02',seq_id+'.txt')
            path_det=os.path.join(path, mode,'det_02',seq_id)
            path_calib=os.path.join(path, mode,'calib',seq_id+'.txt')

            process(impath=path_image,velopath=path_velo,oxtspath=path_oxts,labelpath=path_label,calibpath=path_calib,savepath=save_path)
        else:
            path_image = os.path.join(path, mode, 'image_02', seq_id)
            path_velo = os.path.join(path, mode, 'velodyne', seq_id)
            path_oxts = os.path.join(path, mode, 'oxts', seq_id+'.txt')
            #path_label = os.path.join(path, mode, 'label_02', seq_id) #no in test
            path_det = os.path.join(path, mode, 'det_02', seq_id)
            path_calib = os.path.join(path, mode, 'calib', seq_id+'.txt')

def object_data(path,mode='training'):
    #for i in range(0,tracking_train_seq+1):
        #seq_id=str(i).zfill(4)
        if mode=='training':
            if not os.path.exists(os.path.join(path, mode,'front_view_label')):
                os.mkdir(os.path.join(path, mode,'front_view_label'))

            save_path=os.path.join(path, mode,'front_view_label')
            path_image=os.path.join(path, mode,'image_2')
            path_velo=os.path.join(path, mode,'velodyne')
            path_label=os.path.join(path, mode,'label_2')
            #path_det=os.path.join(path, mode,'det_2')
            path_calib=os.path.join(path, mode,'calib')

            process_object(impath=path_image,velopath=path_velo,labelpath=path_label,calibpath=path_calib,savepath=save_path)
        else:
            path_image = os.path.join(path, mode, 'image_2')
            path_velo = os.path.join(path, mode, 'velodyne')
            #path_label = os.path.join(path, mode, 'label_2', seq_id) #no in test
            #path_det = os.path.join(path, mode, 'det_2', seq_id)
            path_calib = os.path.join(path, mode, 'calib')

def verify_tracking(path,mode='training'):
    for i in range(1,tracking_train_seq+1):
        seq_id=str(i).zfill(4)
        if mode=='training':
            if not os.path.exists(os.path.join(path, mode,'front_view_label')):
                os.mkdir(os.path.join(path, mode,'front_view_label'))

            save_path=os.path.join(path, mode,'front_view_label',seq_id)
            path_image=os.path.join(path, mode,'image_02',seq_id)
            path_velo=os.path.join(path, mode,'velodyne',seq_id)
            path_oxts=os.path.join(path, mode,'oxts',seq_id+'.txt')
            path_label=os.path.join(path, mode,'label_02',seq_id+'.txt')
            path_det=os.path.join(path, mode,'det_02',seq_id)
            path_calib=os.path.join(path, mode,'calib',seq_id+'.txt')

            verify_process(impath=path_image,velopath=path_velo,oxtspath=path_oxts,labelpath=path_label,calibpath=path_calib,savepath=save_path)
        else:
            path_image = os.path.join(path, mode, 'image_02', seq_id)
            path_velo = os.path.join(path, mode, 'velodyne', seq_id)
            path_oxts = os.path.join(path, mode, 'oxts', seq_id+'.txt')
            #path_label = os.path.join(path, mode, 'label_02', seq_id) #no in test
            path_det = os.path.join(path, mode, 'det_02', seq_id)
            path_calib = os.path.join(path, mode, 'calib', seq_id+'.txt')

def verify_object(path,mode='training'):
    if mode=='training':
        save_path=os.path.join(path, mode,'front_view_label')
        path_image=os.path.join(path, mode,'image_2')
        path_label=os.path.join(path, mode,'label_2')
        #print(save_path)
        #print(path_image)
        #print(path_label)
        num=len(os.listdir(path_image))
        for idx in range(0,num):
            img=cv2.imread(path_image+'/'+str(idx).zfill(6)+'.png')
            #print(path_image+'/'+str(idx).zfill(6)+'.png')
            [h,w,c]=img.shape
            label_old=os.path.join(path_label,str(idx).zfill(6)+'.txt')
            label_new=os.path.join(save_path,str(idx).zfill(6)+'.txt')
            file = open(label_old, 'r')
            old = file.readlines()
            file.close()

            for o in old:
                #print(o)
                o=o.strip('\n').split(' ')
                xmin = int(float(o[4]))
                ymin = int(float(o[5]))
                xmax = int(float(o[6]))
                ymax = int(float(o[7]))
                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0,), 2)

            file = open(label_new, 'r')
            new = file.readlines()
            file.close()

            for n in new:
                n=n.strip('\n').split(' ')
                xmid = float(n[1])
                ymid = float(n[2])
                width = float(n[3])
                height = float(n[4])
                xmin = int((xmid - width / 2) * w)
                xmax = int((xmid + width / 2) * w)
                ymin = int((ymid - height / 2) * h)
                ymax = int((ymid + height / 2) * h)
                img = cv2.rectangle(img, (xmin + 5, ymin + 5), (xmax - 5, ymax - 5), (0, 0, 255,), 2)
            cv2.imshow("",img)
            cv2.waitKey()
    else:
        path_image = os.path.join(path, mode, 'image_02')
        path_velo = os.path.join(path, mode, 'velodyne')
        path_det = os.path.join(path, mode, 'det_02')
        path_calib = os.path.join(path, mode, 'calib')

def verify_process(impath,oxtspath,velopath,labelpath,calibpath,savepath):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    #single frame do not need imu, imu contains pose calculation
    imglist=os.listdir(impath) # total frame
    tframe=len(imglist)
    labellist=load_tracking_label(labelpath)
    for idx in range(0,tframe):
        #output_label(savepath, str(idx).zfill(6), None, None)
        img=cv2.imread(os.path.join(impath,str(idx).zfill(6))+'.png')
        [h,w,c]=img.shape
        for j in labellist:
            if j[0][-1]==str(idx):
                for cl in j:
                    #img=img
                    img=cv2.rectangle(img,(cl[3],cl[4]),(cl[5],cl[6]),(255,0,0,),2)
        #print(os.path.exists(os.path.join(impath,str(idx).zfill(6))+'.png'))
        newlabelpath=os.path.join(impath,str(idx).zfill(6)).replace('image_02','front_view_label')+'.txt'
        #print(newlabelpath)
        file=open(newlabelpath,'r')
        newlabels=file.readlines()
        #print(len(newlabels))
        file.close()
        for nl in newlabels:
            nl=nl.strip('\n').split(' ')
            xmid=float(nl[1])
            ymid=float(nl[2])
            width=float(nl[3])
            height=float(nl[4])
            xmin=int((xmid-width/2)*w)
            xmax=int((xmid+width/2)*w)
            ymin = int((ymid - height / 2) * h)
            ymax = int((ymid + height / 2) * h)
            #print((xmin, ymin), (xmax, ymax))
            img = cv2.rectangle(img, (xmin+5, ymin+5), (xmax-5, ymax-5), (0, 0, 255,), 2)
        cv2.imshow("",img)
        cv2.waitKey()

def process_object(impath,velopath,labelpath,calibpath,savepath):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    imglist=os.listdir(impath) # total frame
    tframe=len(imglist)
    #velolist=os.listdir(velopath)
    for idx in range(0,tframe):
        img=cv2.imread(os.path.join(impath,str(idx).zfill(6)+'.png'))
        csavepath=os.path.join(savepath,str(idx).zfill(6)+'.txt')
        #print(csavepath)
        #'''
        f_new_label=open(csavepath,'w')
        [h,w,c]=img.shape
        clabel=os.path.join(labelpath,str(idx).zfill(6)+'.txt')
        f_label=open(clabel,'r')
        cl=f_label.readlines()
        f_label.close()
        for l in cl:
            l=l.strip('\n').split(' ')
            #print(l)
            cls=l[0]
            coolusion=l[2] # 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
            if cls!='DontCare' and cls!='Misc' and (coolusion=='0' or coolusion=='1'):
                angle=float(l[3])
                #xmin = int(float(l[4]))
                #xmax = int(float(l[5]))
                #ymin = int(float(l[6]))
                #ymax = int(float(l[7]))
                xmid = str((float(l[4]) + float(l[6])) / 2 / w)
                ymid = str((float(l[5]) + float(l[7])) / 2 / h)
                width = str(abs(float(l[6]) - float(l[4])) / w)
                height = str(abs(float(l[7]) - float(l[5])) / h)
                coord=xmid+' '+ymid+' '+width+' '+height
                #print(coord)
                #f_new_label.write(str(original_class_dic.get(l[0]))+' '+coord+'\n') # 6 classes
                f_new_label.write(str(original_class_dic.get(l[0])) + ' ' + coord + '\n') #3 classes

        f_new_label.close()

def process(impath,oxtspath,velopath,labelpath,calibpath,savepath):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    #single frame do not need imu, imu contains pose calculation
    imglist=os.listdir(impath) # total frame
    tframe=len(imglist)
    #calib=load_calib(calibpath)
    imu=load_oxts(oxtspath)
    velolist=os.listdir(velopath)
    labellist=load_tracking_label(labelpath)
    #print(len(imglist),len(velolist),len(labellist))
    #iter according to image list, create empty files
    for idx in range(0,tframe): #
        output_label(savepath, str(idx).zfill(6), None, None)
    # iter according to label list for non-empty labels
    for idx in labellist:#range(0,tframe):
        #print(idx)
            #print('i:',i)
        #print(os.path.join(impath, idx[0][-1].zfill(6)+'.png'))
        im = cv2.imread(os.path.join(impath, idx[0][-1].zfill(6)+'.png'))
        #print('input:',savepath, idx[0][-1].zfill(6), idx, im.shape)
        output_label(savepath, idx[0][-1].zfill(6), idx, im.shape)

def transcoord(l,s):
    xmid=str(float(l[3]+l[5])/2/s[1])
    ymid=str(float(l[4]+l[6])/2/s[0])
    width=str(abs(l[5]-l[3])/s[1])
    height=str(abs(l[6]-l[4])/s[0])
    return xmid+" "+ymid+" "+width+" "+height

def output_label(savepath,img_id,llist,size):

    filename=os.path.join(savepath,img_id+'.txt')
    if llist==None or size==None:
        f = open(filename, 'w')
        f.close()
    else:
        #print(filename)
        #print(llist)
        #'''
        f=open(filename,'w')
        if llist != 'empty':
            for l in llist:
                coord=transcoord(l,size)
                #print(original_class_dic.get(l[0]),coord) # 0=car 1=people
                #print(combine_class_dic.get(l[0]), coord) # 0=car 1=people 2=cyclist
                #f.write(str(original_class_dic.get(l[0]))+' '+coord+'\n')# 6 classes
                f.write(str(combine_class_dic.get(l[0]))+' '+coord+'\n') #3 classes
        f.close()
        #'''


def load_oxts(path):
    assert os.path.exists(path)
    imu=utils.load_oxts_packets_and_poses(path)


def load_object_label(path):
    assert os.path.exists(path)
    with open(path, 'r') as f:
        labels = []
        last_frame = 0
        current = []
        for line in f.readlines():
            line=line.split(' ')
            if line[1]=='-1' and line[2]=='DontCare':# ignored objects
                continue
            if line[4]=='2' or line[4]=='3': #0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
                continue
            #print(line)
            frame_id=line[0]
            cal = line[2]
            occlusion = int(float(line[3]))
            angle = float(line[5])
            x1 = int(float(line[6]))
            y1 = int(float(line[7]))
            x2 = int(float(line[8]))
            y2 = int(float(line[9]))
            if frame_id==last_frame:
                #if current:
                current.append([cal,occlusion,angle,x1,y1,x2,y2,frame_id])
            else:
                if current:
                    labels.append(current)
                    current=[]
                #else:
                #    labels.append('empty')
                last_frame=frame_id
                current.append([cal, occlusion, angle, x1, y1, x2, y2,frame_id])
        if current:
            labels.append(current)
    #print(labels[426])
    return(labels)

def load_tracking_label(path):
    assert os.path.exists(path)
    with open(path, 'r') as f:
        labels = []
        last_frame = 0
        current = []
        for line in f.readlines():
            line=line.split(' ')
            if line[2]=='DontCare' or line[2]=='Misc': #remove ignored objects
                continue
            if line[4]=='2' or line[4]=='3': #remove occlude objects
                continue

            #print(line)
            frame_id=line[0]
            cal = line[2]
            occlusion = int(float(line[3]))
            angle = float(line[5])
            x1 = int(float(line[6]))
            y1 = int(float(line[7]))
            x2 = int(float(line[8]))
            y2 = int(float(line[9]))
            if frame_id==last_frame:
                #if current:
                current.append([cal,occlusion,angle,x1,y1,x2,y2,frame_id])
            else:
                if current:
                    labels.append(current)
                    current=[]
                #else:
                #    labels.append('empty')
                last_frame=frame_id
                current.append([cal, occlusion, angle, x1, y1, x2, y2,frame_id])
        if current:
            labels.append(current)
    #print(labels[426])
    return(labels)

def load_calib(path):
    """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
    # convert it to a namedtuple to prevent it from being modified later
    data = {}

    # Load the calibration file
    #calib_filepath = os.path.join(self.sequence_path + '.txt', 'calib.txt')
    filedata = utils.read_calib_file(path)

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(filedata['P0'], (3, 4))
    P_rect_10 = np.reshape(filedata['P1'], (3, 4))
    P_rect_20 = np.reshape(filedata['P2'], (3, 4))
    P_rect_30 = np.reshape(filedata['P3'], (3, 4))

    data['P_rect_00'] = P_rect_00
    data['P_rect_10'] = P_rect_10
    data['P_rect_20'] = P_rect_20
    data['P_rect_30'] = P_rect_30

    # Compute the rectified extrinsics from cam0 to camN
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
    data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
    data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
    data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
    data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

    # Compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    # Compute the stereo baselines in meters by projecting the origin of
    # each camera frame into the velodyne frame and computing the distances
    # between them
    p_cam = np.array([0, 0, 0, 1])
    p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
    p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
    p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
    p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

    data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
    data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)  # rgb baseline
    return data

# generate labels

tracking_data(os.path.join(filedir,model[1]))

#verify labels
#verify_tracking(os.path.join(filedir,model[1]))

#object_data(os.path.join(filedir,model[0]))
#verify_object(os.path.join(filedir,model[0]))
