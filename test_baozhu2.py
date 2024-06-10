import os
from mycommon.utils import *
import qiefen
import torch
import test_mdzx_pic
import test_mdzx_heibai_pic
import test_cls_pic
import threading
import queue
import random
from HCNetSDK import *
from PlayCtrl import *
import socket
import json
from test_HCNetCam import *

from get_config import *

from client import Client
from myflask import *

if config['haikang']:
    from MultipleCamerasDemo import *
draw_out = test_mdzx_pic.draw_out
# from test_mdzx_heibai_pic import *ttd
from concurrent.futures import ThreadPoolExecutor
import datetime

#from mytest_Automation import *
# TODO ADD BY LEI
from loguru import logger
logger.add(".\\log\\file_{time}.log",rotation="1 week")


'''if config['webcamera']:
    from vlcplayer import *'''

if config['wangkoutcp']:
    from modbusreadwrite import *
    tcpmodbus = ModbusAssistant2()

if config['moni']:
    import DHCamera_my as DHCamera
elif config['daheng']:
    import DHCamera
elif config['dalsa']:
    from Dalsa import *

if config['IOctr']:
    if config['IOxinghao'] == 'PCI-1750' or config['IOxinghao'] == 'PCIE-1730':
        print("IO 文件加载")
        from test_Automation import *
else:
    print("虚拟IO 文件加载")
    from mytest_Automation import *

class ImgCam:
    @staticmethod
    def getDeviceNum():
        return 1

    def __init__(self, pa):
        self.li = listdir(pa, ['.jpg', '.png', '.bmp'])
        self.li = [f'{pa}/{x}' for x in self.li]
        self.i = 0

    def open(self, x):
        self.i = 400
        if len(self.li) > 0:
            return 1
        return 0

    def close(self):
        self.i = 0
        return 0

    def stop(self):
        self.i = 0
        return 0

    def getFrame(self):
        if self.i >= len(self.li):
            self.i = 0

        if self.i < len(self.li):
            # print(f'getFrame {self.i} {self.li[self.i]}')
            im = cv_imread(self.li[self.i], 1)
            self.i += 1
            return im

        return None


def get_cam_thread(ui):
    pass
    '''mdzx_pic = get_mdzx_pic()

    def cam_thread():
        while (ui.cam_thread_runing):
            imgs = [ui.cam_li[i].getFrame() for i in range(len(ui.cam_li))]
            ui.imgSignal.emit(imgs)
            mdzx_pic(imgs[0])
            time.sleep(0.001)'''


def action103tt(img0, id):
    # print(f'{id} action103')
    out = qiefen.detect_a0(img0)
    # stepSignal.emit(str(out), data)

    # ng = 0
    for i in out:
        img = imrect(img0, i)
        # img = cv2.cvtColor(numpy.asarray(i), cv2.COLOR_RGB2BGR)
        im1, end1 = test_mdzx_pic.mdzx_pic(img)

        # print(f'.............caise:{end1}....................')
        if int(end1) != 0:
            ng = 1

    # print(f'{id} action103 end')
    return id


from PIL import Image, ImageDraw, ImageFont


# 用于给图片添加中文字符
def draw_text_cn(img, text, pos, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否为OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        r'C:/Windows/Fonts/simsun.ttc', textSize, encoding="utf-8")  # 中文字体
    draw.text((int(pos[0]), int(pos[1])), text, textColor, font=fontText)  # 写文字
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def action104tt(img0, id):
    # print(f'{id} action104')
    # print(f'.................4:{img0.shape}..........')
    if len(img0.shape) == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    out = qiefen.detect_a0(img0)
    # ng = 0
    for i in out:
        # img = cv2.cvtColor(numpy.asarray(i), cv2.COLOR_RGB2BGR)
        img = imrect(img0, i)
        im1, end1 = test_mdzx_heibai_pic.mdzx_pic(img)
        t = end1
        # print(f'..........heibai:{t}...................')

    # print(f'{id} action104 end')
    return id

# TODO ADD BY LEI
from test_yolov8 import *
yolo_obb_model = get_model(model_dict,'chilun')
yolo_obb_model('.\\Image_20240430163636566.jpg')
logger.info('yolov8模型加载')

def get_yolov8_detect_obb():

    def detect(img,r0=None,post_fix = ''):
        # TODO ADD BY LEI
        print(f' IN LINE 160 [model infer]')
        obb_boxes,conf,cls = yolo_obb_model(img)
        if r0 is not None:
            pass
        out = []
        for rect, lab, conf in zip(obb_boxes,cls,conf):
            out.append([rect, lab, conf])
        return img,out
    return detect

# def get_yolov8_detect_obb():
#     def cal_length(img0,id,config):
#         print(f'IN LINE 172')
#         cc = []
#         return id, img0, cc
#     return cal_length

def get_cal_length():
    def cal_length(img0,id,config):
        logger.info('宽度计算')
        cc = []
        return id, img0, cc
    return cal_length



def get_actions():
    # TODO CHANGED BY LEI 模型调用
    action101 = get_yolov8_detect_obb()
    action102 = get_cal_length()
    # action102 = test_cls_pic.get_action('kuaiwei', '2')
    # action103 = test_mdzx_pic.get_action(5, 5, '')
    # action104 = test_mdzx_pic.get_action(5, 5, '2')
    ff = [
        # action103,
        # action104,
        action101,
        action102
    ]
    return ff
    # return []
# TODO END ==== ==== ==== ====

def get_actions1():
    tt = 180 / 1000

    def action101(img0, id, config):
        cc = []
        time.sleep(tt)
        return id, img0, cc

    def action102(img0, id, config):
        cc = []
        time.sleep(tt)
        return id, img0, cc

    def action101t(img0, id, config):
        cc = []
        return id, img0, cc

    def action102t(img0, id, thd):
        cc = []
        return id, img0, cc

    def action103(img0, id, config):
        cc = []
        time.sleep(tt)
        return id, img0, cc

    def action104(img0, id, config):
        cc = []
        time.sleep(tt)
        # print(f'{id} action104 end')
        return id, img0, cc

    ff = [
        action101,
        action102,
        action103,
        action104
    ]
    return ff


def time_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    return t0

#action101, action102, action103, action104 = get_actions()
myaction = get_actions()
#Camera1suanfa
#if config['baozutest']:
# action_map= {}
action_map = {'quexian1': myaction[int(config['Camera1suanfa'])], 'quexian2': myaction[int(config['Camera2suanfa'])],'length': myaction[int(config['Camera3suanfa'])], }
'''elif config['saiwangtest']:
    action_map = {'kuaijian': action103, 'kuaiwei': action103, 'caise': action103, 'heibai': action103}
elif config['shougongzuoye']:
    action_map = {'kuaijian': action103, 'kuaiwei': action103, 'caise': action103, 'heibai': action103}
else:
    action_map = {'kuaijian': action103, 'kuaiwei': action103, 'caise': action103, 'heibai': action103}'''


def get_outp(index):
    d = get_date_stamp()
    if index == 1:
        outp = f'{cam1ok}/{d}'
    elif index == 2:
        outp = f'{cam2ok}/{d}'
    elif index == 3:
        outp = f'{cam3ok}/{d}'
    elif index == 4:
        outp = f'{cam4ok}/{d}'
    elif index == 5:
        outp = f'{cam5ok}/{d}'
    elif index == 6:
        outp = f'{cam6ok}/{d}'
    elif index == 7:
        outp = f'{cam7ok}/{d}'
    elif index == 8:
        outp = f'{cam8ok}/{d}'
    elif index == 9:
        outp = f'{cam1ng}/{d}'
    elif index == 10:
        outp = f'{cam2ng}/{d}'
    elif index == 11:
        outp = f'{cam3ng}/{d}'
    elif index == 12:
        outp = f'{cam4ng}/{d}'
    elif index == 13:
        outp = f'{cam5ng}/{d}'
    elif index == 14:
        outp = f'{cam6ng}/{d}'
    elif index == 15:
        outp = f'{cam7ng}/{d}'
    elif index == 16:
        outp = f'{cam8ng}/{d}'
    mkdir(outp)
    return outp


class BoundThreadPoolExecutor(ThreadPoolExecutor):
    # 对ThreadPoolExecutor 进行重写，给队列设置边界
    def __init__(self, qsize: int = None, *args, **kwargs):
        super(BoundThreadPoolExecutor, self).__init__(*args, **kwargs)
        self._work_queue = queue.Queue(qsize)


def action100_null(img0, id, config):
    logger.info('未运行算法')
    cc = []
    return id, img0, cc


def action_all(data, skip1):
    index, numpy_image, t01, xx = data
    # action_fun = action100_null if skip1 else action_map[xx[1]]
    logger.info('>>>>>>>>>>xx[1]:{x}',x=xx[1])
    action_fun = action100_null if xx[1] not in action_map.keys() else action_map[xx[1]]
    # action_fun = action_map[xx[1]]
    config['skip'] = skip1
    logger.info('actions:{a}',a=action_fun)
    # if index in [1, 2]:
    #    return [index, numpy_image, numpy_image, []]

    t0 = time.time()

    if config['chongqinmeidi']:
        if config['webcamera']:
            if index == 5 or index == 6:
                im1 = numpy_image
                tt = []
            else:
                sp = numpy_image.shape
                # c = numpy_image.channels()
                # print(f'>>>>>img  shape:{sp} len sp:{len(sp)}')
                if len(sp) == 2:
                    numpy_image = gray2bgr(numpy_image)
                id, im1, tt = action_fun(numpy_image, index, config)
        else:
            sp = numpy_image.shape
            # c = numpy_image.channels()
            # print(f'>>>>>img  shape:{sp} len sp:{len(sp)}')
            if len(sp) == 2:
                numpy_image = gray2bgr(numpy_image)
            id, im1, tt = action_fun(numpy_image, index, config)
    else:
        if config['donotmodel']:
            im1 = numpy_image
            tt = []
        else:
            sp = numpy_image.shape
            # c = numpy_image.channels()
            #print(f'>>>>>img  shape:{sp} len sp:{len(sp)}')
            if len(sp) == 2:
                numpy_image = gray2bgr(numpy_image)
            logger.info('action_fun start')
            id, im1, tt = action_fun(numpy_image, index, config)
            logger.info('action_fun end::{i}',i=index)
    t1 = time.time()
    print("action_all end:",index)
    return [index, numpy_image, im1, tt, xx, t01]


def get_output():
    pa = 'D:/data/220329竹筷'
    pa = pa if os.path.exists(pa) else 'E:/data/220329竹筷'
    outpa = f'{pa}/out'
    return outpa


post_pool = BoundThreadPoolExecutor(qsize=8, max_workers=8)


def all_quexian(cc):
    out = []
    for i in range(len(cc)):
        c, r0 = cc[i]
        isng = False
        for r, la, conf in c:
            thd = config[la]
            if conf > thd[1]:
                isng = True
            if isng:
                out.append([r, la, conf])

    return out


def ret_action_all_post(data, ss):
    index, numpy_image, im1, tt, xx, t0, cnt = data
    print("tt4:",tt)
    # chufa(index, tt, config)
    _, name = xx
    try:
        im1 = draw_out(numpy_image, tt, config)
    except Exception as e:
        #print(f'draw_out error {e}')
        pass
    data = [index, numpy_image, im1, tt, xx, t0, cnt]
    if 1:
        if ss is not None:
            ss.showSignal.emit(data)
            ss.showSignal2.emit(data)
        else:
            imshow(f'im{index}_{name}', im1, 1)
            pass
        # draw_out(imgs[j], )
        # cv_imwrite(fn, cc[1])
        # fn = f'{outpa}/{i:05d}_{j}_img.jpg'
    if 0:
        outpa = get_output()
        mkdir(f'{outpa}/out{index}')
        fn = f'{outpa}/out{index}/{cnt:05d}_org.jpg'
        cv_imwrite(fn, im1)

    okflag = True
    for t in tt:
        quexian_type = 'OK'
        for x in t:
            myindex = int(t[1][3] / 132)
            if myindex >= 5:
                continue
            for y in x:
                if isinstance(y, list):
                    name = y[1]
                    score1 = y[2]
                    c, thd = config[name]
                    if score1 > thd and name != 'OK':
                        quexian_type = name
                        # print(f'quexian_type:{quexian_type} score:{score1} r:{y[0]}')
        if quexian_type != 'OK':
            okflag = False

    if config['saveimg'] and len(tt) != 0:
        if okflag == False:
            if config['ngimgsave']:
                outp = get_outp(index)
                s = get_time_stamp()
                pp = f'{outp}/img_{index}_{s}'
                if config['imgbmpsave']:
                    fn2 = f'{pp}_img2.bmp'
                else:
                    fn = f'{pp}_img.jpg'
                    fn2 = f'{pp}_img2.jpg'
                if config['ngimgret']:
                    fn = f'{pp}_ret.jpg'
                    cv_imwrite(fn, im1)

                #cv_imwrite(fn2, numpy_image)
                cv2.imwrite(fn2, numpy_image)
        else:
            if config['okimgsave']:
                outp = get_outp(index + 8)
                s = get_time_stamp()
                pp = f'{outp}/img_{index}_{s}'
                if config['imgbmpsave']:
                    fn2 = f'{pp}_ret2.bmp'
                else:
                    fn2 = f'{pp}_ret2.jpg'
                cv2.imwrite(fn2, numpy_image)
                # cv_imwrite(fn2, numpy_image)

def ret_action_all_post_v2(data, ss, myskip):
    # TODO CHANGED BY LEI
    index, numpy_image, im1, tt, xx, t0, cnt = data
    # chufa(index, tt, config)
    #_, name = xx
    logger.info('ret_action_all_post_v2>>1:{index}',index=index)
    if config['tcpsend'] and index <=4:
        print(f'>>>>tcpsend start:{index}')
        tcpflag = True
        for mii in tt[0][0]:
            print(f'tcpsend mii:{mii}')
            if mii[2] < 60:
                continue
            # 构建字典数据
            data = {
                'channel': index,
                'category': mii[1],
                'coordinates': mii[0],
                'score': mii[2]
            }
            # 将字典转换为JSON格式
            json_data = json.dumps(data)
            # 发送数据到服务器
            tcpsenkehu[0].sendall(json_data.encode())

            try:
                # 接收服务器响应
                response = tcpsenkehu[0].recv(1024)
            except socket.timeout:
                # 如果超时时间内未收到消息，则进行相应处理
                print(f'等待超时，未收到消息:{index}')
            # 打印服务器响应
            print(f'服务器响应{index}:{response.decode()}')
            mystr = str(response.decode())
            indexsn[index] = mystr
            if config['showsn']:
                pass
                #print(f'SNnumber::{mystr}')
                '''sn111 = mystr["code"]
                print(f'sn111:{sn111}')
                sn222 = sn111.split(";")
                print(f'sn2:::::{sn222}')'''
                #ss.SNnumber.setText(str(mystr))
                #print(f'ss.SNnumber.setText>>')
            if 'NG' in mystr:
                tcpflag = False
                if config['lianji']:
                    webcampic[3] =True

        print(f'>>>>tcpflag:{tcpflag} :{index}')
        if config['chongqinmeidi'] and index <=4:
            h, w = numpy_image.shape[:2]
            r = [0,0,w,h]
            color = (0, 255, 0) if tcpflag else (0, 0, 255)
            drawrect(numpy_image, r, color, 2)
            if tcpflag:
                text = 'OK'
            else:
                text = 'NG'
                datastatic_ng[index][0] += 1
            s1 = 5
            cv2.putText(numpy_image, text, (int(w/2), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, s1, color, s1 + 1)
    print(f'ret_action_all_post_v2>>2:{index}')
    try:
        im1 = draw_out(numpy_image, tt, config)
    except Exception as e:
        #print(f'draw_out error {e}')
        pass
    h,w = im1.shape[:2]
    resizerate = 1
    if w > 1920:
        resizerate = w/1920
    if resizerate != 1:
        im2 = cv2.resize(im1, (int(w / resizerate), int(h / resizerate)))
        if config['saiwangtest']:#拼图功能
            #im3 = cv2.resize(im1, (int(w / 4), int(h / 4)))
            #im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
            #print("im2 shape:",im2.shape)
            bigimg[0] = np.concatenate((bigimg[0], im2), axis=0)
            #print("bigimg = concatenate")
        data = [index, numpy_image, im2, tt, xx, t0, cnt]
    else:
        data = [index, numpy_image, im1, tt, xx, t0, cnt]
    if 1:
        if ss is not None:
            ss.showSignal.emit(data)
            ss.showSignal2.emit(data)
        else:
            #imshow(f'im{index}_{name}', im1, 1)
            pass
        # draw_out(imgs[j], )
        # cv_imwrite(fn, cc[1])
        # fn = f'{outpa}/{i:05d}_{j}_img.jpg'
    if 0:
        outpa = get_output()
        mkdir(f'{outpa}/out{index}')
        fn = f'{outpa}/out{index}/{cnt:05d}_org.jpg'
        cv_imwrite(fn, im1)

    okflag = True
    for i in range(len(tt)):
        c, r0 = tt[i]
        for r, la, conf in c:
            if la in config.keys():
                c, thd = config[la]
                if conf > thd:
                    okflag = False

    #print(f'okflagokflag:{okflag}')

    if config['flasksend']:
        url = 'http://127.0.0.1:5000/send_images'  # 替换为正确的 IP 和端口
        mydata = {}
        mydata['images'] = '111'
        mydata['product'] = imgframe_ir[index-1][1]
        mydata['class'] = 1
        mydata['end'] = 'ok'

        response = requests.post(url, data=mydata)
        print("sned flask data ret:",response.json())

    #print("tttttt:",tt[0][0])


    ttflag = True
    '''if config['baozutest']:
        if len(tt) == 0:
            ttflag = False'''
    if config['donotmodel']:
        ttflag = True
        okflag = True

    if config['saveimg'] and ttflag:
        if okflag == False:
            outp = get_outp(index+8)
            s = get_time_stamp()
            if config['saiwangtest']:
                bianhao = config['biaohao']
            else:
                bianhao = ''
            pp = f'{outp}/img{bianhao}_{index}_{s}'
            fn = f'{pp}_img_ng.jpg'
            fn2 = f'{pp}_img2.jpg'
            fn3 = f'{pp}_img2.bmp'
            #cv_imwrite(fn2, numpy_image)
            if config['ngimgsave']:
                if config['imgbmpsave']:
                    cv2.imwrite(fn3, numpy_image)
                else:
                    cv2.imwrite(fn2, numpy_image)
            if config['ngimgret']:
                cv_imwrite(fn, im1)
            print(f'>>>>>>>>>>>save ng img')
        else:
            if config['okimgsave']:
                outp = get_outp(index)
                s = get_time_stamp()
                if config['saiwangtest']:
                    bianhao = config['biaohao']
                else:
                    bianhao = ''
                pp = f'{outp}/img{bianhao}_{index}_{s}'
                fn = f'{pp}_ret.jpg'
                fn2 = f'{pp}_ret2.jpg'
                fn3 = f'{pp}_ret2.jpg'
                #cv_imwrite(fn, im1)
                # print("name:",fn2)
                #if myskip == 1:
                #print("NG path:",fn2)
                if config['imgbmpsave']:
                    cv2.imwrite(fn3, numpy_image)
                else:
                    cv2.imwrite(fn2, numpy_image)
                # cv_imwrite(fn2, numpy_image)
                print(f'>>>>>>>>>>>save ok img')

def checkbox_output(cc, config,img):
    if config['baozutest']:
        for i in range(len(cc)):
            c, r0 = cc[i]
            j = 0
            for r, la, conf in c:
                if la == 'daojiao':
                    cc[i][0][j][1] = 'daojiao2'
                    la = 'daojiao2'
                elif la == 'suojie2':
                    if r[0] > 250 and r[0] < 375:
                        cc[i][0][j][2] = 1.02
                    elif r[0] > 1126:
                        cc[i][0][j][2] = 1.02
                        # print(f'rect suojie:{r} i:{i} j:{j}')
                elif la == 'suojie':
                    if r[0] > 225 and r[0] < 360:
                        cc[i][0][j][2] = 1.02
                    elif r[0] > 1073:
                        cc[i][0][j][2] = 1.02
                elif la == 'lasi' or la == 'lasi2' or la == 'ljlasi' or la == 'ljlasi2':
                    if r[0] > 450:
                        #cc[i][0][j][2] = cc[i][0][j][2] - 10
                        cc[i][0][j][2] = cc[i][0][j][2]
                elif la == 'wanqu2':
                    if cc[i][0][j][2] == 100:
                        cc[i][0][j][2] = 1.0
                elif la == 'kt' or la == 'kt2' or la == 'kw' or la == 'kw2':
                    h = r[2] - r[0]
                    w = r[3] - r[1]
                    D = min(h,w)
                    D = D/3
                    print(f'r:{r} D:{D} cc:{cc}')
                    #cc[i][0][j][2] = D
                    if la == 'kw':
                        la = 'kt'
                        cc[i][0][j][1] = 'kt'
                        print(">>>>>>1")
                    elif la == 'kw2':
                        la = 'kt2'
                        cc[i][0][j][1] = 'kt2'
                        print(">>>>>>2")

                    cc[i][0][j][2] = D
                    if la == 'kt':
                        if D > config[la][1] + config['fanwei'][1]/3 or D < config[la][1] - config['fanwei'][1]/3:
                            cc[i][0][j][2] = D + 500
                            print(f'1kt D:{D}')
                        else:
                            cc[i][0][j][2] = D - config['fanwei'][1]/3
                            fw = config['fanwei'][1]/3
                            print(f'2kt D:{D} score:{cc[i][0][j][2]} fanwei:{fw} config:{config[la][1]}')
                    elif la == 'kt2':
                        if D > config[la][1] + config['fanwei2'][1]/3 or D < config[la][1] - config['fanwei2'][1]/3:
                            cc[i][0][j][2] = D + 500
                        else:
                            cc[i][0][j][2] = D - config['fanwei2'][1] / 3
                            fw = config['fanwei2'][1] / 3
                            #print(f'kt2 D:{D} fanwei:{fw}')

                elif la == 'wanqu':
                    if cc[i][0][j][2] == 100:
                        cc[i][0][j][2] = 1.0
                elif la == 'heidian' or la == 'heidian2':
                    img_heidian = img[r[1]:r[3], r[0]:r[2]]
                    gray = cv2.cvtColor(img_heidian, cv2.COLOR_BGR2GRAY)
                    #seeds = originalSeed(gray, th=253)
                    seedMark = regionGrow(gray, thresh=8, p=8)
                    index_yi = np.where(seedMark > 1)
                    mean = np.mean(gray[index_yi])
                    mean = mean / 2.55
                    seedMark = 255 - seedMark
                    index_yi = np.where(seedMark > 1)
                    mean2 = np.mean(gray[index_yi])
                    mean2 = mean2 / 2.55
                    mydis = abs(mean - mean2)
                    if math.isnan(mydis):
                        mydis = 1
                    mydis = int(mydis*3)+15
                    area = r[2]*r[3]
                    cc[i][0][j][2] = area * mydis/10000
                    #print(f'mean:{mean} mean2:{mean2} dis:{mydis} area * mydis:{cc[i][0][j][2]}')
                    myname = f'ldc{la}'

                    #cc[i][0].append([[r[0],r[1]+50,r[2],r[3]],myname,mydis])

                #print(f'la:{la}')
                statu, thd = config[la]
                if statu == 0:
                    cc[i][0][j][2] = 1.02
                    #print(f'la:{la} c:{c} thd:{thd} thd2:{cc[i][0][j][2]}')
                j += 1


def ret_action_all_post_v3(tt, myskip, numpy_image, index):
    okflag = False
    for t in tt:
        quexian_type = 'OK'
        for x in t:
            myindex = int(t[1][3] / 132)
            if myindex >= 5:
                continue
            for y in x:
                if isinstance(y, list):
                    name = y[1]
                    score1 = y[2]
                    c, thd = config[name]
                    if score1 > thd and name != 'OK':
                        quexian_type = name
                        # print(f'quexian_type:{quexian_type} score:{score1} r:{y[0]}')
        if quexian_type != 'OK':
            okflag = True

    if 1:
        if okflag == False:
            outp = get_outp(index)
            s = get_time_stamp()
            pp = f'{outp}/img_{index}_{s}'
            fn = f'{pp}_img.jpg'
            fn2 = f'{pp}_img2.jpg'
            # cv_imwrite(fn, im1)
            #cv_imwrite(fn2, numpy_image)
            #if myskip == 1:
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ok path:{fn2} tt:{tt}')
            cv2.imwrite(fn2, numpy_image)
        else:
            outp = get_outp(index + 4)
            s = get_time_stamp()
            pp = f'{outp}/img_{index}_{s}'
            fn = f'{pp}_ret.jpg'
            fn2 = f'{pp}_ret2.jpg'
            #cv_imwrite(fn, im1)
            # print("name:",fn2)
            #if myskip == 1:
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ng name:{fn2}')
            #print("NG path:",fn2)
            #cv2.imwrite(fn2, numpy_image)
            cv_imwrite(fn2, numpy_image)


def result_rect(la,conf,src,r):
    im1 = src.copy()
    text = f'{la} {conf:.1f}'
    s = 1
    color = (0, 0, 255)
    drawrect(im1, r, color, 2)
    cv2.putText(im1, text, (int(r[0]), int(r[1])), cv2.FONT_HERSHEY_SIMPLEX, s, color, s + 1)
    return im1

def camera_output(index,put,ss):
    if config['dancamjgcl']:
        if config['modbusset']:
            chufareturnnum[index] += 1
            if index == 1:
                index_reg = 10
                #if not config['moni']:
                logger.info('cam index:{index} put :{put}',index=index,put=put)
                logger.info('write to {r}',r=index_reg)
                rec = ss.mydo.write_register(index_reg,put)
                return rec
            elif index == 2:
                index_reg = 12
                #if not config['moni']:
                print(f'LEI LINE 765')
                logger.info('cam index:{index} put :{put}',index=index,put=put)
                logger.info('write to {r}',r=index_reg)
                rec = ss.mydo.write_register(index_reg, put)
                return rec
            elif index == 3:
                index_reg = 14
                #if not config['moni']:
                print(f'LEI LINE 765')
                logger.info('cam index:{index} put :{put}',index=index,put=put)
                logger.info('write to {r}',r=index_reg)
                rec = ss.mydo.write_register(index_reg, put)
                return rec


def tongyongchufa(index, cc,config,src,ss):
    resultok = True
    nntime2 = time.time()
    distime = nntime2 - reoldtime[index]
    if distime < 0.9:
        slpt = 0.9 - distime
        time.sleep(slpt)
        print(f'延时等待通信{slpt}')
    for i in range(len(cc)):
        c, r0 = cc[i]
        j = 0
        for r, la, conf in c:
            if la in config.keys():
                c, thd = config[la]
                if conf > thd:
                        resultok = False
    if not resultok:
        datastatic_ng[index][0] += 1
        rec = camera_output(index,2,ss)
    else:
        rec = camera_output(index,1,ss)
    rec = str(rec)
    if "OK" in rec:
        
        logger.info('通讯成功:{rec}',rec=rec)
    else:
        
        logger.info('通讯失败:{rec},即将重启串口',rec=rec)
        print(f'LEI LINE 799')
        ss.mydo.myclose()
        ss.mydo.restartctr()
    nntime = time.time()
    jiangetime = nntime - reoldtime[index]
    if jiangetime < 0.6:
        print(f'结果反馈间隔小于0.6')
    print(f'do result>>>>>>>>>>>>>>end {index}:{indexnum[index]} {resultok} 结果反馈间隔:{nntime - reoldtime[index]}')
    reoldtime[index] = nntime

def result_chufa(index, cc,config,src):
    if config['duozhangjgcl'] and modbustatu1[0] == 0 and modbustatu1[5] == 0:
        time.sleep(0.5)
        if qq[0].qsize() == 0:
            if config['saiwangtest']:  # 拼图功能
                outp = get_outp(index + 8)
                s = get_time_stamp()
                if config['saiwangtest']:
                    bianhao = config['biaohao']
                else:
                    bianhao = ''
                pp = f'{outp}/img{bianhao}_{index}_{s}'
                fn = f'{pp}_all.jpg'
                fn2 = f'{pp}_allstart.jpg'
                myww = (int)(16384/2)
                r = [0,0,int(myww/(myww/1920)),int(myww/(myww/1920))]
                img = imrect(bigimg[0], r)
                testimg = img.copy()
                testimg = cv2.resize(testimg, (640, 640))
                testimg = testimg.astype(np.uint8)
                id, im1, cc = action103(testimg, 3, config)
                print("tt:",cc)
                ###############################
                minpixy = 640
                madd = 0
                for i in range(len(cc)):
                    c, r0 = cc[i]
                    j = 0
                    for r, la, conf in c:
                        if la in config.keys():
                            if la == 'start' or la == 'end':
                                if r[1] < minpixy:
                                    minpixy = r[1]
                                if la == 'start':
                                    madd = 40
                                elif la == 'end':
                                    madd = 30
                                print(f'la:{la} r:{r}')
                        j += 1
                print("minpixy:",minpixy)
                ###############################
                cv2.imwrite(fn2, testimg)
                cv2.imwrite(fn, bigimg[0])
                bigimg[0] = np.zeros((10, int(myww/(myww/1920)),3))
            modbustatu1[1] = 0

            if len(mutilimgresult) > 0:
                if config['wangkoutcp']:
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>NG:",mutilimgresult)
                    for i in range(len(mydata)):
                        mydata.clear()
                    #12600000
                    imgdis = 1260/63
                    pixdis = imgdis/2000
                    fangda = (16384/2) / 640
                    print("minpixypix:", minpixy*fangda)
                    print(f'imgdis:{imgdis} pixdis:{pixdis}')
                    for r, la, conf, ma,imgone in mutilimgresult:
                        #mmm = (r[1]*pixdis)+ma*imgdis
                        if ma > 0 and ma < 63 and la != 'end' and la != 'start':
                            if config['saiwangtest']:
                                if minpixy == 640:
                                    minpixy = 0
                                mmm = (ma * 2000 + r[1] - int(minpixy*fangda)) * pixdis
                            else:
                                mmm = (ma*2000+r[1])*pixdis
                            mmm = round(mmm, 3)
                            mmm = mmm + madd
                            mmm = mmm*1000
                            if mmm > 15000:
                                mmm = mmm
                                print(f'第{ma}张图,{int(r[1])}像素位置,{mmm}')
                                mydata.append(mmm)
                                myresultimg.append(imgone)
                    for i in range(len(mutilimgresult)):
                        mutilimgresult.clear()
                    if len(mydata) > 0:
                        modbustatu1[4] = 2
                    else:
                        modbustatu1[4] = 1
            else:
                if config['wangkoutcp']:
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>OK:",mutilimgresult)
                    modbustatu1[4] = 1

    a,b,c = src[0,0]
    for i in range(len(cc)):
        c, r0 = cc[i]
        j = 0
        for r, la, conf in c:
            if la in config.keys():
                thd = config[la]
                if conf > thd[1]:
                    if config['duozhangjgcl']:
                        print(f'>>第{a}张图,{la}')
                        r1 = r[0] - 320
                        if r1 < 0:
                            r1 = 0
                        r2 = r[1] - 240
                        if r2 < 0:
                            r2 = 0
                        r3 = r[0] + 320
                        if r3 > 8191:
                            r3 = 8191
                        r4 = r[1] + 240
                        if r4 > 2000:
                            r4 = 1999
                        r_ng = [r1, r2, r3, r4]
                        im1 = result_rect(la, conf, src, r)
                        img = imrect(im1, r_ng)
                        mutilimgresult.append([r,la,conf,a,img])
                    elif config['danzhangjgcl']:
                        pass
            j += 1

def doresult(index, cc, config):#统计结果
    for i in range(len(cc)):
        c, r0 = cc[i]
        j = 0
        for r, la, conf in c:
            if la in config.keys():
                c, thd = config[la]
                if conf > thd:
                    for n in range(len(mytestnamelabel[index-1])):
                        if mytestnamelabel[index-1][n][0] == la:
                            mytestnamelabel[index - 1][n][1] += 1
    return True

def ret_action_all(data, ss):
    index, numpy_image, im1, tt, xx, t0, cnt = data
    print("ret_action_all1:", index)
    checkbox_output(tt, config,numpy_image)
    doresult(index,tt, config)
    #print("ret_action_all2:",tt)
    #draw_output(tt, numpy_image)
    data = [index, numpy_image, im1, tt, xx, t0, cnt]
    _, name = xx
    #post_pool.submit(ret_action_all_post, data, ss)
    myskip = qq[2].qsize()

    post_pool.submit(ret_action_all_post_v2, data, ss,myskip)

    if len(tt) > 0 and not config['donotmodel']:
        if config['chukuai']:
            post_pool.submit(chufa, index, tt, config)
        elif config['saiwangtest']:
            post_pool.submit(result_chufa, index, tt,config,numpy_image)
        else:
            post_pool.submit(tongyongchufa, index, tt,config, numpy_image, ss)
        #chufa(index, tt, config)
        #print("tt3:",tt)
        #chufa_v2(index, tt, config, cam_flagimg)
        #post_pool.submit(chufa_v2, index, tt, config,cam_flagimg)
    else:
        if config['chukuai']:
            if index == 1:
                moutkuaizi2[5] += 1
            elif index == 2:
                moutkuaizi[5] += 1
    t1 = time.time()
    t1t0 = t1 - t0
    print(f'all time {index} {t1t0:.5f} {int(config["skip"])} {qq[0].qsize()} {qq[1].qsize()}')
    logging.info(f'all time {index} {t1t0:.5f} {int(config["skip"])} {qq[0].qsize()} {qq[1].qsize()}')
    return [index, numpy_image, im1, tt, xx]


qq = [queue.Queue(100) for i in range(8)]
cam_flagimg = [0,0,0,0,0,0,0,0]


# retqq = [queue.Queue() for i in range(4)]


def test_baozhu2(ss=None, qqi=[0, 1, 2, 3,4, 5, 6, 7]):
    outpa = get_output()
    mkdir(outpa)
    outpa1 = f'{outpa}/out'
    mkdir(outpa1)
    os.system(f'del {outpa}\\*.jpg'.replace('/', '\\'))
    # [os.system(f'del {outpa}\\out{0}\\*.jpg'.replace('/', '\\')) for i in tti]

    pool = BoundThreadPoolExecutor(qsize=8, max_workers=8)
    # retpool = BoundThreadPoolExecutor(qsize=8, max_workers=8)
    # retq = queue.Queue()
    # retret = retpool.submit(result_thread, retq, ss, tti, outp)
    cnt = 0
    skip_cnt = 0
    # os.system(f'del {outpa1}\\*.jpg'.replace('/', '\\'))
    while True:
        # qq = queue.Queue()
        datas = [qq[i].get() for i in qqi]
        if datas[0] is None:
            break

        cnt += 1
        rets = []
        t0 = time_synchronize()
        skip = False
        # print(f'{pool._work_queue.qsize()}')
        # print(f'qq size {qq[-1].qsize()}')
        modbustatu1[5] = qq[0].qsize()
        #print(">>>>>>>>>>>>>>>qq[0].qsize():",qq[3].qsize())
        if config['isskip']:
            if qq[0].qsize() > 1:
                # action_f = action100_null
                skip_cnt += 1
                skip = True

        # print(f'qq len = {qq.qsize()} {pool._work_queue.qsize()} {cnt} {skip_cnt}')
        # qq = queue.Queue()
        # ss.qq.get(timeout)
        # tt = [2]
        for data in datas:
            
            index, numpy_image, t01, xx = data
            logger.info('>>>>>>>>data index {i}',i=index)
            logger.info('>>>>>>>>data xx {x}',x=xx)
            ret = pool.submit(action_all, data, skip)
            
            rets.append(ret)
            # ss.retqq[index-1].put([ret, cnt, t0])
        ccc = [f.result() for f in rets]
        # print(ccc)
        t1 = time_synchronize()
        t1t0 = t1 - t0
        # retq.put([rets, cnt, t0])
        #print(">>>>>>>5")
        for j in range(len(ccc)):
            ret_action_all(ccc[j] + [cnt], ss)
        # t1 = time_synchronize()
        # t1t02 = t1-t0
        # print(f'all time {t1t0} {t1t02}')

    # retq.put(None)
    # retret.result()
    # ret.add_done_callback(print_func)
    return 0

def cam_callback(ss, index, img):
    chufanum[index] += 1
    if circlecheckflag[0] == False:
        return
    ntime = time.time()
    print(f'callback {index}:{indexnum[index]} 触发间隔:{ntime - oldtime[index]}')
    indexnum[index] += 1
    oldtime[index] = ntime
    if config[f'Camera{index}'][0] != 0:
        w = config[f'Camera{index}'][0]
        h = config[f'Camera{index}'][1]
        img = cv2.resize(img, (w, h))
    # cams, i = ss[0]
    i = index - 1
    cc = config['cams']
    #print(">>>>>cc:",cc)
    xx = [cc[x] for x in range(len(cc)) if cc[x][0] == index]

    testimg = img.copy()

    '''if index == 4:
        cam_flagimg[3] = 1
        mytime1 = datetime.datetime.now()
        #print("cam_flagimg[3] = 1:",mytime1)
    elif index == 3:
        cam_flagimg[2] = 1'''
    t0 = time.time()
    if len(xx) > 0:
        #print("readHoldingRegisters modbustatu1:",modbustatu1[0])
        if config['moni']:
            modbustatu1[0] = 1
        if config['duozhangjgcl']:
            if modbustatu1[0]:
                #print("img num:",modbustatu1[1])
                testimg[0, 0] = modbustatu1[1]
                testimg[0, 1] = modbustatu1[1] + 1
                testimg[0, 2] = modbustatu1[1] + 2
                modbustatu1[1] += 1
                qq[i].put([index, testimg, t0, xx[0]])
                #print(">>>>>put")
        else:
            qq[i].put([index, testimg, t0, xx[0]])
        pass

def startflasksend():
    flasksend_app.run()

def WebCamGetHcFrameThread(fun,ss,index,webcameraLi):
    print("WebCamGetHcFrameThread:",index)
    pool = BoundThreadPoolExecutor(qsize=8, max_workers=8)
    pool.submit(HcGetframe, fun, ss,index,webcameraLi)

def HcGetframe(fun, ss,index,webcameraLi):
    print("HcGetframe:",index)
    while True:
        if config['webcameraonepice']:
            if config['chongqinmeidi'] and config['cameranum'] > 4:
                myind = index - 4
            else:
                myind = index
            if webcampic[0]:
                webcampic[0] = False
                time.sleep(config['yanshi1text1'])
                frame = webcameraLi[myind - 1].getFrame()
                if len(frame) != 1:
                    fun(ss, index, frame)
                    time.sleep(0.1)
            elif webcampic[2]:
                webcampic[2] = False
                time.sleep(config['yanshi2text2'])
                frame = webcameraLi[myind - 1].getFrame()
                if len(frame) != 1:
                    fun(ss, index, frame)
                    time.sleep(0.1)
        else:
            frame = webcameraLi[index - 1].getFrame()
            if len(frame) != 1:
                fun(ss, index, frame)
                time.sleep(0.1)
def openctpsned():
    print('openctpsned')
    # 定义服务器地址和端口
    server_address = config['tcpsendipadresstext']
    server_port = int(config['tcpsendduankoutext'])
    try:
        # 创建TCP客户端套接字
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 检查连接状态
        if client_socket.fileno() != -1:
            print("连接成功！")
        else:
            print("连接失败！")
        # 连接服务器
        ret = client_socket.connect((server_address, server_port))
        # 设置超时时间为5秒
        client_socket.settimeout(0.3)
        tcpsenkehu.append(client_socket)
        print(f'openctpsned:{ret}')
    except ZeroDivisionError:
        # 捕获 ZeroDivisionError 异常并处理
        print("异常并处理！")

class Baozhu:
    def __init__(self) -> None:
        self.pool = BoundThreadPoolExecutor(qsize=10, max_workers=10)
        self.ret = []

    def start(self, ss):
        if circlecheckflag[0] == False:
            circlecheckflag[0] = True
            circlecheckflag[1] = False
        elif circlecheckflag[0] and circlecheckflag[1]:
            filename2 = f'./testtt.jpg'
            imgcaise = cv2.imread(filename2, 1)
            if len(imgcaise) > 0:
                myaction[int(config['Camera1suanfa'])](imgcaise, 3, config)
                myaction[int(config['Camera2suanfa'])](imgcaise, 3, config)
                print("预处理模型")


            if config['flasksend']:
                self.pool.submit(startflasksend)
                print("开启flask 客户端")

            if config['tcpsend']:
                print("开启tcpsend 客户端")
                self.pool.submit(openctpsned)

            if int(config['tcpnumtext']) > 0:
                for i in range(int(config['tcpnumtext'])):
                    pa = f'TCP{i+1}'
                    paduankou = f'TCPduankou{i+1}'
                    print(f'pa:{config[pa]} paduankou:{config[paduankou]} {type(config[paduankou])}')
                    client = Client()
                    client.connect_init(i,config[pa], int(config[paduankou]))
            t1flg = config['wangkoutcp']
            t2falg = config['moni']
            print(f't1flg:{t1flg} t2falg:{t2falg}')
            if config['wangkoutcp']:
                tcpmodbus = ModbusAssistant2()
                ret = tcpmodbus.opentcpModbus(config['ipadresstext'],config['duankoutext'],'Modbus TCP')
                print("opentcpModbus:",ret)
                logging.info(f'opentcpModbus:{ret}')
                if ret:
                    tcpmodbus.single_loop_send_noui(500)

                #if config['chuankou']:
            #if not config['moni']:
            print(f'打开串口modbus')
            print(f'LEI: 打开串口 调用startctr')
            rec = ss.mydo.startctr()
            logging.info(f'打开串口modbus:{rec}')
            #ss.mydo.myclose()
            #ss.mydo.restartctr()


            if config['daheng'] or config['moni']:
                # ss 窗口句柄
                # tti 相机配置
                nums = DHCamera.GetCamNum()  # 获取相机列表
                self.cams = [DHCamera.DHCamera(i + 1) for i in range(nums)]

                for i, camera in enumerate(self.cams):
                    modbustatu1[6] += 1
                    index = i + 1
                    camera.OpenDevice()  # 打开一个相机
                    camera.StartContinusAcqCallback(cam_callback, [self.cams, i])  # 设置回调
                    print(f"open camera {index}")

                if config['wangkoutcp'] and config['moni']:
                    tcpmodbus.writesingelrigster(1, 651, 1)
            elif config['dalsa']:
                dalsagetFrameThread(cam_callback,ss)
                if config['wangkoutcp']:
                    tcpmodbus.writesingelrigster(1, 651, 1)
            elif config['haikang']:
                # 海康相机
                global deviceList
                deviceList = MV_CC_DEVICE_INFO_LIST()
                global tlayerType
                tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
                global obj_cam_operation
                obj_cam_operation = 0
                # global b_is_run
                # b_is_run = False
                global nOpenDevSuccess
                nOpenDevSuccess = 0
                global devList
                enum_devices()
                open_device()
                # set_triggermode()
                # TODO ADD BY LEI
                logger.info('相机触发')
                start_grabbing(cam_callback, self)

            if config['webcamera']:
                print("config['webcamera'] ok")
                webcameravlc = False
                webcameraHC = True
                '''if webcameravlc:
                    for i in range(int(config['cameranum'])-myi):

                        c1 = Player()

                        if i == 0:
                            sr1 = config['ipadresstext1']
                            con1 = config['duankoutext1']
                        elif i == 1:
                            sr1 = config['ipadresstext2']
                            con1 = config['duankoutext2']
                        elif i == 2:
                            sr1 = config['ipadresstext3']
                            con1 = config['duankoutext3']
                        elif i == 3:
                            sr1 = config['ipadresstext4']
                            con1 = config['duankoutext4']
                        ipdrre = f'{sr1}/{con1}'
                        ip1_port_user_pw = 'rtsp://%s:%s@%s' % ('admin', 'a12345678', ipdrre)  # vlc调用
                        c1.play(ip1_port_user_pw, i+1)
                        ss.webcameraLi.append(c1)
                        #WebCamGetFrameThread(cam_callback,ss,i+1,ss.webcameraLi)
                        WebCamGetFrameThread(cam_callback, ss, myi, ss.webcameraLi)
                        myi += 1'''
                if webcameraHC:
                    mydec = 0
                    for i in range(int(config['cameranum'])):
                        if i == 0:
                            sr1 = config['ipadresstext1']
                            con1 = config['duankoutext1']
                        elif i == 1:
                            sr1 = config['ipadresstext2']
                            con1 = config['duankoutext2']
                        elif i == 2:
                            sr1 = config['ipadresstext3']
                            con1 = config['duankoutext3']
                        elif i == 3:
                            sr1 = config['ipadresstext4']
                            con1 = config['duankoutext4']
                        url = f'{sr1} {con1} admin a12345678'


                        # WebCamGetHcFrameThread(cam_callback,ss,i+1,ss.webcameraLi)
                        if config['chongqinmeidi'] and config['cameranum'] > 4:
                            if i == 0 or i == 1:
                                print(f'url:{url} index:{i + 1}')
                                c1 = HCNetCam(url)
                                ss.webcameraLi.append(c1)
                                WebCamGetHcFrameThread(cam_callback, ss, i+1+4, ss.webcameraLi)
                        else:
                            c1 = HCNetCam(url)
                            ss.webcameraLi.append(c1)
                            print(f'url:{url} index:{i + 1}')
                            WebCamGetHcFrameThread(cam_callback, ss, i + 1, ss.webcameraLi)


            if len(self.ret) == 0:
                if 1:
                    self.ret = [self.pool.submit(test_baozhu2, ss, [i]) for i in range(config['cameranum'])]
                else:
                    self.ret = [self.pool.submit(test_baozhu2, ss, [0, 1, 2, 3])]

            #if not config['moni']:
            ss.mydo.write_register(0, 1)


    def stop(self):
        #ss.mydo.write_register(0, 2)
        circlecheckflag[0] = False
        '''if config['tcpsend']:
            # 关闭客户端套接字连接
            tcpsenkehu[0].close()'''
        '''if len(self.ret) > 0:
            if config['daheng'] or config['moni']:
                for camera in self.cams:
                    camera.CloseDevice()  # 打开一个相机
            elif config['daheng']:
                dalsacircle = False
            [q.put(None) for q in qq]
            [f.result() for f in self.ret]
            self.ret = []'''


def test_baozhu2_thd(ss=None):
    baozhu = Baozhu()
    # tti = [[3, 'caise']]
    baozhu.start(ss)
    for i in range(300):
        time.sleep(1)
    baozhu.stop()
    return 0


if __name__ == "__main__":
    pa = 'D:/data/220329竹筷/pic'
    pa = 'D:/data/220329竹筷'
    #test_baozhu2_thd()
    # cv2.destroyAllWindows()

