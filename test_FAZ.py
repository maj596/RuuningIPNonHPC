import tensorflow as tf
import numpy as np
import os
import scipy.misc as misc
import natsort
import time
import utils
import lossfunc
from options.test_options import TestOptions
import model

def main(argv=None):
    TP_list = []
    TN_list = []
    FP_list = []
    FN_list = []
    opt = TestOptions().parse()
    test_results = os.path.join(opt.saveroot,'test_results')
    utils.check_dir_exist(test_results)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    DATA_SIZE = opt.data_size.split(',')
    DATA_SIZE = [int(DATA_SIZE[0][1:]),int(DATA_SIZE[1]),int(DATA_SIZE[2][:-1])]
    BLOCK_SIZE = opt.block_size.split(',')
    BLOCK_SIZE = [int(BLOCK_SIZE[0][1:]),int(BLOCK_SIZE[1]),int(BLOCK_SIZE[2][:-1])]
    label_path = os.path.join(opt.dataroot,opt.mode,'label')
    label_names = natsort.natsorted(os.listdir(label_path))

    x=tf.placeholder(tf.float32, shape=[None] + BLOCK_SIZE + [opt.input_nc+1], name="input_image")
    y=tf.placeholder(tf.int32, shape=[None, 1, BLOCK_SIZE[1], BLOCK_SIZE[2], 1], name="annotation")
    y_,pred_, variables,sf= model.IPN(x=x,PLM_NUM=opt.PLM_num, LAYER_NUM=opt.layer_num,NUM_OF_CLASS=opt.NUM_OF_CLASS)
    model_loss = lossfunc.cross_entropy(y_,y)

    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    restore_path = os.path.join('FAZ_logs','best_model',natsort.natsorted(os.listdir(os.path.join('FAZ_logs','best_model')))[-1])
    o_itr = natsort.natsorted(os.listdir(restore_path))[-1][11:-5]
    saver.restore(sess, os.path.join(restore_path,'model.ckpt-'+o_itr))
    print("Model restored...")

    test_images= np.zeros((1, BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[2], opt.input_nc+1))
    cube_images= np.zeros((1, BLOCK_SIZE[0], DATA_SIZE[1], DATA_SIZE[2], opt.input_nc+1))
    test_annotations = np.zeros((1,1,BLOCK_SIZE[1],BLOCK_SIZE[2],1))

    modalitylist = os.listdir(os.path.join(opt.dataroot,opt.mode))
    modalitylist = natsort.natsorted(modalitylist)
    print(modalitylist)

    result = np.zeros((DATA_SIZE[1], DATA_SIZE[2]))

    cubelist = os.listdir(os.path.join(opt.dataroot, opt.mode,modalitylist[0]))
    cubelist = natsort.natsorted(cubelist)

    # for kk,cube in enumerate(cubelist):
    #     loss2 = 0
    #     bscanlist = os.listdir(os.path.join(opt.dataroot, opt.mode, modalitylist[0], cube))
    #     bscanlist=natsort.natsorted(bscanlist)
    #     for i,bscan in enumerate(bscanlist):
    #         for j,modal in enumerate(modalitylist):
    #             if modal!="label":
    #                 cube_images[0,:,:,i,j]=np.array(misc.imresize(misc.imread(os.path.join(opt.dataroot,opt.mode,modal,cube,bscan)),[BLOCK_SIZE[0], DATA_SIZE[1]], interp='nearest'))
    #         cube_images[0, :, :, i, j] = np.array(misc.imresize(misc.imread(os.path.join('FAZ_logs','distancemap', bscan)),[BLOCK_SIZE[0], DATA_SIZE[1]], interp='nearest'))

    #     for i in range(DATA_SIZE[1] // BLOCK_SIZE[1]):
    #         for j in range(0, DATA_SIZE[2] // BLOCK_SIZE[2]):
    #             test_images[0, 0:BLOCK_SIZE[0], 0:BLOCK_SIZE[1], 0:BLOCK_SIZE[2], :] = cube_images[0, :,BLOCK_SIZE[1] * i:BLOCK_SIZE[1] * (i + 1), BLOCK_SIZE[2] * j:BLOCK_SIZE[2] * (j + 1), :]
    #             score,result0,piece_loss,sf0 = sess.run([y_,pred_,model_loss,sf], feed_dict={x: test_images,y: test_annotations})
    #             result[BLOCK_SIZE[1] * i:BLOCK_SIZE[1] * (i + 1), BLOCK_SIZE[2] * j:BLOCK_SIZE[2] * (j + 1)] = sf0[0, 0, :,:,1] * 255
    loss_sum = 0
    acc_sum = 0
    dice_sum = 0
    for kk,cube in enumerate(cubelist):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        loss2 = 0
        bscanlist = os.listdir(os.path.join(opt.dataroot, opt.mode,modalitylist[0], cube))
        bscanlist=natsort.natsorted(bscanlist)
        for i,bscan in enumerate(bscanlist):
            for j,modal in enumerate(modalitylist):
                if modal!="label":
                    cube_images[0,:,:,i,j]=np.array(misc.imresize(misc.imread(os.path.join(opt.dataroot,opt.mode,modal,cube,bscan)),[BLOCK_SIZE[0], DATA_SIZE[1]], interp='nearest'))
            cube_images[0, :, :, i, j] = np.array(misc.imresize(misc.imread(os.path.join('FAZ_logs','distancemap', bscan)),[BLOCK_SIZE[0], DATA_SIZE[1]], interp='nearest'))
        for i in range(DATA_SIZE[1] // BLOCK_SIZE[1]):
            for j in range(0, DATA_SIZE[2] // BLOCK_SIZE[2]):
                test_images[0, 0:BLOCK_SIZE[0], 0:BLOCK_SIZE[1], 0:BLOCK_SIZE[2], :] = cube_images[0, :,BLOCK_SIZE[1] * i:BLOCK_SIZE[1] * (i + 1), BLOCK_SIZE[2] * j:BLOCK_SIZE[2] * (j + 1), :]
                score,result0,piece_loss,sf0 = sess.run([y_,pred_,model_loss,sf], feed_dict={x: test_images,y: test_annotations})
                result[BLOCK_SIZE[1] * i:BLOCK_SIZE[1] * (i + 1), BLOCK_SIZE[2] * j:BLOCK_SIZE[2] * (j + 1)] = sf0[0, 0, :,:,1] * 255
    
        # label = misc.imread(os.path.join(label_path,label_names[kk])) * 255
        # for i in range(result.shape[0]):
        #     for j in range(result.shape[1]):
        #         if result[i, j] > 127.5 and label[i, j] > 127.5:
        #             TP += 1
        #         elif result[i, j] < 127.5 and label[i, j] < 127.5:
        #             TN += 1
        #         elif result[i, j] > 127.5 and label[i, j] < 127.5:
        #             FP += 1
        #         elif result[i, j] < 127.5 and label[i, j] > 127.5:
        #             FN += 1

        
        for num in range(1,20):
            nx = int(np.random.normal(DATA_SIZE[1]/2, 50))
            ny = int(np.random.normal(DATA_SIZE[2]/2, 50))
            mx = int(BLOCK_SIZE[1]/2)
            my = int(BLOCK_SIZE[2]/2)
            if nx<=BLOCK_SIZE[1]/2 or nx>=DATA_SIZE[1]-mx:
                nx=int(DATA_SIZE[1]/2)
            if ny<=BLOCK_SIZE[2]/2 or ny>=DATA_SIZE[2]-my:
                ny=int(DATA_SIZE[2]/2)
            test_images[0, 0:BLOCK_SIZE[0], 0:BLOCK_SIZE[1], 0:BLOCK_SIZE[2], :] = cube_images[0, :,(nx-mx):(nx+mx),(ny-my):(ny+my), :]
            score, result0, piece_loss, sf0 = sess.run([y_, pred_, model_loss, sf],
                                                       feed_dict={x: test_images, y: test_annotations})
            result[(nx-mx):(nx+mx),(ny-my):(ny+my)] =result[(nx-mx):(nx+mx),(ny-my):(ny+my)]/2+ sf0[0, 0, :,:, 1] * 255/2
            
        
        
        
        result = np.where(result > 127.5, 255, result)
        result = np.where(result < 127.5, 0, result)
        label = misc.imread(os.path.join(label_path,label_names[kk]))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i, j] > 0 and label[i, j] == 255:
                    TP += 1
                elif result[i, j] < 255 and label[i, j] == 255:
                    FN += 1
                elif result[i, j] > 0 and label[i, j] == 0:
                    FP += 1
                else:
                    TN += 1
                    

    
        # print the confusion matrix
        print('Confusion matrix for cube', cube)
        print('     Predicted')
        print('     |  0  |  1  |')
        print('-----------------')
        print('True |', TN, '|', FP, '|')
        print('     |-----|-----|')
        print('     |', FN, '|', TP, '|') 
        TP_list.append(TP)
        TN_list.append(TN)
        FP_list.append(FP)
        FN_list.append(FN)
        
        
        
        
        loss2 = loss2 / (DATA_SIZE[1] // BLOCK_SIZE[1]) * (DATA_SIZE[2] // BLOCK_SIZE[2]) #changed / to *
        label = misc.imread(os.path.join(label_path,label_names[kk])) * 255
        acc = utils.cal_acc(result,label)
        dice = utils.cal_Dice(result,label)
        print(cube,'loss -> {:.3f}, acc -> {:.3f}, dice -> {:.3f}'.format(loss2,acc,dice))
        loss_sum += loss2
        acc_sum += acc
        dice_sum += dice        
        
        
        
        # Apply thresholding to cube_images
        # cube[cube > 0] = 255
        print("Saved image: ", cube)
        misc.imsave(os.path.join(test_results,cube+"_FAZ_pre.bmp"), result.astype(np.uint8))
    print('')
    print('mean: ','loss -> {:.3f}, acc -> {:.3f}, dice -> {:.3f}'.format(loss_sum/len(label_names),acc_sum/len(label_names),  \
         dice_sum/len(label_names)))
    print ("TP_list",TP_list)
    print ("TN_list",TN_list)
    print ("FP_list",FP_list)
    print ("FN_list",FN_list)



if __name__ == "__main__":
    tf.app.run()
