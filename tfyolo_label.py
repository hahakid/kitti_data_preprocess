import os
import cv2


def process_tracking(mode):
    image_path='../'+mode+'/training/image_02' #image path
    yolo_label_path='../'+mode+'/training/front_view_label' # front view label path
    pyyolo_label_path='../'+mode+'/tfyolo/pylabel.txt' #only one file of output
    class_path='../'+mode+'/tfyolo/class.names' #names file


    #original_class_dic = {0:'Car',1:'Van',2:'Truck', 3:'Pedestrian', 4:'Person_sitting', 5:'Cyclist',6:'Tram'}
    combine_class_dic = {0:'vehicle',1:'person',2:'cyclist'}

    seqlist=os.listdir(yolo_label_path)

    if not os.path.exists(pyyolo_label_path):
        f=open(pyyolo_label_path,'w')
        f.close()

    pylabelfile=open(pyyolo_label_path,'w')

    for s in seqlist:
        cp=os.path.join(yolo_label_path,s)
        llist=os.listdir(cp)
        for l in llist:
            cl=os.path.join(cp,l)
            f=open(cl,'r')
            lines=f.readlines()
            f.close()
            #print(type(lines))
            img=os.path.join(image_path,s,l.replace('txt','png'))
            #print(os.path.exists(img) if os.path.exists(img) else "no file")
            h,w,c=cv2.imread(img).shape
            if len(lines):
                #print(lines)
                labelout=''
                for idx in lines:
                    #if idx[0]=='None':
                    print(s,l)
                    idx=idx.strip('\n').split(' ')
                    cls=int(idx[0]) #original_class_dic.get(int(idx[0]))
                    xmid=float(idx[1])
                    ymid=float(idx[2])
                    width=float(idx[3])
                    height=float(idx[4])
                    coord='%.2f,%.2f,%.2f,%.2f,%d '%((xmid-width/2)*w,(ymid-height/2)*h,(xmid+width/2)*w,(ymid+height/2)*h,int(idx[0])) # xmin ymin xmax ymax class
                    labelout=labelout+coord
                labelout=img+" "+labelout+"\n"
                pylabelfile.write(labelout)

    pylabelfile.close()

    classfile=open(class_path,'w')
    for k in combine_class_dic:
        classfile.write(combine_class_dic[k]+'\n')
    classfile.close()

def process_object(mode):
    image_path='../'+mode+'/training/image_2' #image path
    yolo_label_path='../'+mode+'/training/front_view_label' # front view label path
    pyyolo_label_path='../'+mode+'/tfyolo/pylabel.txt' #only one file of output
    class_path='../'+mode+'/tfyolo/class.names' #names file


    #original_class_dic = {0:'Car',1:'Van',2:'Truck', 3:'Pedestrian', 4:'Person_sitting', 5:'Cyclist',6:'Tram'}
    combine_class_dic = {0:'vehicle',1:'person',2:'cyclist'}

    #seqlist=os.listdir(yolo_label_path)

    if not os.path.exists(pyyolo_label_path):
        f=open(pyyolo_label_path,'w')
        f.close()

    pylabelfile=open(pyyolo_label_path,'w')


    cp=yolo_label_path
    llist=os.listdir(cp)
    for l in llist:
        cl=os.path.join(cp,l)
        f=open(cl,'r')
        lines=f.readlines()
        f.close()
        #print(type(lines))
        img=os.path.join(image_path,l.replace('txt','png'))
        #print(os.path.exists(img) if os.path.exists(img) else "no file")
        h,w,c=cv2.imread(img).shape
        if len(lines):
            #print(lines)
            labelout=''
            for idx in lines:
                if idx[0] == 'None':
                    continue
                    #print(idx,l)
                idx=idx.strip('\n').split(' ')
                cls=int(idx[0]) #original_class_dic.get(int(idx[0]))
                xmid=float(idx[1])
                ymid=float(idx[2])
                width=float(idx[3])
                height=float(idx[4])
                coord='%.2f,%.2f,%.2f,%.2f,%d '%((xmid-width/2)*w,(ymid-height/2)*h,(xmid+width/2)*w,(ymid+height/2)*h,int(idx[0])) # xmin ymin xmax ymax class
                labelout=labelout+coord

            labelout=img+" "+labelout+"\n"
            pylabelfile.write(labelout)

    pylabelfile.close()

    classfile=open(class_path,'w')
    for k in combine_class_dic:
        classfile.write(combine_class_dic[k]+'\n')
    classfile.close()

process_object('object')
process_tracking('tracking')





