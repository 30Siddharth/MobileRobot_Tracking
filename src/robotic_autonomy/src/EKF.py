#!/usr/bin/env python
import numpy as np
import human_walk
import matplotlib.pyplot as plt
import pdb
import rospy
from geometry_msgs.msg import PoseWithCovariance
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist

class EKF():
    def __init__(self,nk,dt,X,U):
        self.nk = nk
        self.dt = 1
        self.X = X
        self.U = U

        self.Sigma_init = np.array([[0.05,0],[0,0.05]])
        self.sigma_measure = np.array([[0.05,0],[0,0.05]])  # <--------<< We can measure this if we want. ****Subscribe from the published topic*****
        self.KalGain = np.random.rand(2,2)

        self.measurement_sub = rospy.Subscriber("/ballxyz",PoseWithCovariance,self.measurement_cb)
        self.z_k = np.array([0,0])
        print(self.z_k) 

        #  Get first prediction
        self.Sx_k_k = self.Sigma_init
        # for k in range(nk):
        #     X_predicted,Sx_k_km1, A = self.prediction(self.X,self.U,self.Sx_k_k)                        # PREDICTION STEP  
        #     self.X_pred = np.concatenate((self.X_pred,X_predicted))                           # For plotting
        #     # self.z_k = rospy.Subscriber("/ballxyz",PoseWithCovariance,self.measurement_cb)       
        #     print(self.z_k)                                                     # Get measurement
        #     X_corrected, self.Sx_k_k = self.correction(X_predicted, Sx_k_km1, self.z_k, self.KalGain)   # CORRECTION STEP 
        #     self.gainUpdate(Sx_k_km1)                  # GAIN UPDATE       
        #     self.X_correc = np.concatenate((self.X_correc,X_corrected))                       # For plotting
        #     self.X = X_corrected  
       
        # self.X_pred = np.zeros([1,3])       # ***** Publish *****
        # self.X_correc = np.zeros([1,3])   # ***** Publish *****
        
        self.pos_pub = rospy.Publisher('/move_base_simple/goal',PoseStamped,queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/move_base/cmd_vel',Twist,queue_size=10)
        self.goalPose = np.array([0.0,0.0])
        self.D_prime = 0.2
        
        
    def update(self):
        self.X_pred = self.X
        self.X_correc = self.X

        for k in range(self.nk):
            X_predicted,Sx_k_km1, A = self.prediction(self.X,self.U,self.Sx_k_k)                        # PREDICTION STEP  
            self.X_pred = np.concatenate((self.X_pred,X_predicted))                           # For plotting      
            # print(self.z_k)                                                     # Get measurement
            X_corrected, self.Sx_k_k = self.correction(X_predicted, Sx_k_km1, self.z_k, self.KalGain)   # CORRECTION STEP 
            self.gainUpdate(Sx_k_km1)                  # GAIN UPDATE       
            self.X_correc = np.concatenate((self.X_correc,X_corrected))                       # For plotting
            self.X = X_corrected  

        self.X_pred = np.reshape(self.X_pred,[6,2])       # ***** Publish *****
        self.X_correc = np.reshape(self.X_correc,[6,2])   # ***** Publish *****


    def prediction(self,x,U,Sigma_km1_km1):
        # Function returns:
        # Predicted State: X_predicted
        # Prediction Sigma: Sigma_k_km1
        # State transition matrix: A

        dt = 1
        X_corrected = x
        noiseSigma_model  = np.array([[0.002,0],[0,0.002]]) # Noise in the motion model
        U[0] = U[0] 
        U[1] = U[1] 
        X_predicted = self.getPose(dt,X_corrected,U) 
        X_predicted = X_predicted + np.random.multivariate_normal([0,0],noiseSigma_model)

        A = self.getGrad(x,U)
        mul1 = np.matmul(A,Sigma_km1_km1)
        At = np.transpose(A)
        mul2 = np.matmul(mul1,At)
        Sigma_k_km1 =   mul2 + noiseSigma_model

        return X_predicted, Sigma_k_km1, A

    def getGrad(self,x,U):
        # Function to return the linearized model
        dl = 0.1
        xs = len(x)
        dx = np.zeros([xs])

        f1 = np.zeros([xs,xs])
        f2 = np.zeros([xs,xs])
        for i in range(xs):
            dx = np.zeros([xs])
            dx[i] = dl/2
            x1 = x - dx
            x2 = x + dx       
            f1[:,i] =  self.dotX(x1,U)
            f2[:,i] =  self.dotX(x2,U)
        return (f1+f2)/dl
            

    def dotX(self,x,U):
        # Non linear state transition model
        # R = 0.30
        thk = np.arctan2(x[1],x[0])
        llv = U[0]
        rlv = U[1]

        dxk = (llv + rlv)*0.5*np.cos(thk)
        dyk = (llv + rlv)*0.5*np.sin(thk)
        # dthk = (llv - rlv)/R

        dX = np.array([dxk,dyk])

        return dX

    def getPose(self,dt,X,U):
        
        xkm = X[0]
        ykm = X[1]
        thkm = np.arctan2(ykm,xkm)
        lk = U[0]
        rk = U[1]
        dX = human_walk.differential_walk(xkm,ykm,lk,rk)
        X = X + dt*dX 

        return X


    def correction(self,x_predict, Sx_k_km1, z_k, KalGain):
        C = np.array([[1,0],[0,1]])
        suprise = z_k - np.matmul(C,x_predict)
        X_k_k = x_predict + np.matmul(KalGain,suprise) 
        mul1 = np.eye(2) - np.matmul(KalGain,C)
        # vec = np.array([0.001,0.001,0.001])
        Sx_k_k =  np.matmul(mul1,Sx_k_km1)

        return X_k_k, Sx_k_k 

    def gainUpdate(self, Sx_k_km1):

        C = np.array([[1,0],[0,1]])
        Csig = np.matmul(C,Sx_k_km1)
        CsigCinv = np.matmul(Csig,np.transpose(C))
        # vec = np.array([0.001,0.001])
        mul2 = CsigCinv + self.sigma_measure 
        eig = np.linalg.eigvals(mul2) 
        # if np.
        mul1 = np.matmul(Sx_k_km1,np.transpose(C))
        self.KalGain = np.matmul(mul1,np.linalg.inv(mul2))


    def getGoalPose(self):
        X_goalPose = self.X_correc[-1,:]             # ***** Publish *****
        self.goalPose = X_goalPose
        print('Goal Pose: ',X_goalPose)
        thk = np.arctan2(X_goalPose[1],X_goalPose[0])
        # msg = PoseWithCovariance()
        msg = PoseStamped()
        msg.header.frame_id = "t265_odom_frame"
        msg.pose.position.x = X_goalPose[0]
        msg.pose.position.y = X_goalPose[1]
        msg.pose.position.z = 0
        msg.pose.orientation.z = np.sin(thk/2)
        msg.pose.orientation.w = np.cos(thk/2)
        covar = np.reshape(self.Sx_k_k,(4,1))
        # msg.covariance[0] = covar[0]
        # msg.covariance[8] = covar [3]
        # msg.covariance[15] = covar [8]
        # msg.covariance[15] = 0.0
        self.pos_pub.publish(msg)

    def measurement_cb(self,data):
        zx = data.pose.position.x
        zy = data.pose.position.y
        self.z_k = np.array([zx,zy])
        return np.array([zx,zy])

    # def getRobotPose_cb(self,data):


    def send_cmd_vel(self):
        Kp_v = 0.2
        Kp_theta = 2
        max_v = 0.5
        max_omega = 0.8

        D = np.linalg.norm(self.goalPose)
        theta = np.arctan2(self.goalPose[1],self.goalPose[0])

        # D = np.linalg.norm(self.z_k)
        # theta = np.arctan2(self.z_k[1],self.z_k[0])

        v = -Kp_v*(self.D_prime - D)
        omega = Kp_theta*theta

        cmd = Twist()
        v_capped = np.ndarray.max(np.array([-1*max_v,np.ndarray.min(np.array([v,max_v]))]))
        omega_capped = np.ndarray.max(np.array([-1*max_omega,np.ndarray.min(np.array([omega,max_omega]))]))
        cmd.linear.x = v_capped
        cmd.angular.z =  omega_capped

        print("(v,omega): ({},{})".format(v_capped,omega_capped))
        self.cmd_vel_pub.publish(cmd)


    

if __name__ == '__main__':
    print("here")
    # ROS machinery
    
    rospy.init_node("EKF")
    rate = rospy.Rate(1)
    nk = 5
    dt = 1
    X = np.array([1,1]) # <----------<< This must be obtained by subcribing to the ball postion topic
    U = np.array([0.2,0.2]) # <--------<< We will pick this up from the random distribution like we have been
    filter = EKF(nk,dt,X,U)

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
    # # Define initial conditions
        filter.update()
        filter.getGoalPose()  
        # print('Measured-->',filter.z_k)
        # filter.send_cmd_vel()      
        rate.sleep()
 
    

    # We will measure the state of the vall every 2 seconds (0.5 Hz) 
    # Predict the 5 second look ahead position.
    # Use the predicted postions to generate goal pose for second (1 Hz).