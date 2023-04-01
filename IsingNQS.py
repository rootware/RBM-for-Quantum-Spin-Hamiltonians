import numpy as np
import matplotlib as mp
import pandas as pd

def IsingLocalEnergy(N,IsingState,SPFx,SPFy,J,hx,hy,hz):
    E = -J*np.sum(IsingState[:,0:N-1]*IsingState[:,1:N],axis=1) - J*IsingState[:,N-1]*IsingState[:,0] - hx*SPFx - hy*SPFy - hz*np.sum(IsingState,axis=1)
    
    return E

def TwoDIsingLocalEnergy(N,a,b,IsingState,SPFx,SPFy,J,hx,hy,hz,NS):
    
    TwoDIsingState = IsingState.reshape(NS,a,b)
    
    E = -J*np.sum(TwoDIsingState[:,0:(a-1),:]*TwoDIsingState[:,1:a,:],axis=(1,2)) - J*np.sum(TwoDIsingState[:,(a-1),:]*TwoDIsingState[:,0,:],axis=1) - J*np.sum(TwoDIsingState[:,:,0:(b-1)]*TwoDIsingState[:,:,1:b],axis=(1,2)) - J*np.sum(TwoDIsingState[:,:,(b-1)]*TwoDIsingState[:,:,0],axis=1) - hx*SPFx - hy*SPFy - hz*np.sum(IsingState,axis=1)
    
    return E

def ThreeDIsingLocalEnergy(N,a,b,c,IsingState,SPFx,SPFy,J,hx,hy,hz,NS):
    
    ThreeDIsingState = IsingState.reshape(NS,a,b,c)
    
    E = -J*np.sum(ThreeDIsingState[:,0:(a-1),:,:]*ThreeDIsingState[:,1:a,:,:],axis=(1,2,3)) - J*np.sum(ThreeDIsingState[:,(a-1),:,:]*ThreeDIsingState[:,0,:,:],axis=(1,2)) - J*np.sum(ThreeDIsingState[:,:,0:(b-1),:]*ThreeDIsingState[:,:,1:b,:],axis=(1,2,3)) - J*np.sum(ThreeDIsingState[:,:,(b-1),:]*ThreeDIsingState[:,:,0,:],axis=(1,2)) - J*np.sum(ThreeDIsingState[:,:,:,0:(c-1)]*ThreeDIsingState[:,:,:,1:c],axis=(1,2,3)) - J*np.sum(ThreeDIsingState[:,:,:,(c-1)]*ThreeDIsingState[:,:,:,0],axis=(1,2)) - hx*SPFx - hy*SPFy - hz*np.sum(IsingState,axis=1)
    
    return E

def HeisenbergLocalEnergy(N,Jx,Jy,Jz,hx,hy,hz,SPFx,SPFy,SPFxx,SPFyy,HeisenbergState):    
    
    E = -Jz*np.sum(HeisenbergState[:,0:(N-1)]*HeisenbergState[:,1:N],axis=1) - Jz*HeisenbergState[:,(N-1)]*HeisenbergState[:,0]
    E = E - hz*np.sum(HeisenbergState,axis=1)
    E = E - hx*SPFx - hy*SPFy - Jx*SPFxx - Jy*SPFyy
    
    return E 

def TrainIsingNQS(J,hx,hy,hz,N,alpha,eps,EpochNum,EpochSize,NumBurn,Skip,eta,reg):
    
    IsingNQS = NeuralQuantumState(N,alpha,eps)
    
    M = alpha*N
    Energy = []
    EnergyVariance = []
    
    for Epoch in range(EpochNum):
        
        PhysicalSamples, SampleThetas = IsingNQS.SampleDistribution(EpochSize,NumBurn,Skip)
        
        #PhysicalSamples = IsingNQS.generate_samples(EpochSize)[0]
        #SampleThetas = IsingNQS.ComputeAllThetas(PhysicalSamples)
                
        SPFx, SPFy = IsingNQS.SumPsiFracs(PhysicalSamples,SampleThetas)
        E = IsingLocalEnergy(N,PhysicalSamples,SPFx,SPFy,J,hx,hy,hz)
                
        aD, bD = IsingNQS.Gradients(PhysicalSamples,SampleThetas)
            
        MaD = np.mean(aD,axis=0)
        MbD = np.mean(bD,axis=0)
        #MWD = np.mean(WD,axis=0)
        MWD = 2*np.dot(aD.T,bD)/EpochSize
                
        #MEaD = np.mean(E[:,np.newaxis]*aD,axis=0)
        #MEbD = np.mean(E[:,np.newaxis]*bD,axis=0)
        #MEWD = np.mean(E[:,np.newaxis,np.newaxis]*WD,axis=0)
        
        MEaD = np.dot(E,aD)/EpochSize
        MEbD = np.dot(E,bD)/EpochSize
        MEWD = 2*np.dot(np.multiply(E,aD.T),bD)/EpochSize
        
        Eng = np.mean(E)
        Evar = np.var(E)
        Energy.append(Eng)
        EnergyVariance.append(Evar)
        
        Ga = MEaD - Eng*MaD
        Gb = MEbD - Eng*MbD
        GW = MEWD - Eng*MWD
        
        IsingNQS.GradientDescentStep(Ga,Gb,GW,eta,reg)
        
        #print('%d. E=%.4f, E_loc = %.4f E_var=%.4f' % (Epoch+1,Eng,Eng/N,Evar))
    
    #ax = pd.Series(Energy).plot(title=r'Local energy $\frac{<E_{loc}>}{N}$',grid=True)
    #ax.set_ylabel(r'$<E_{loc}>/N$')
    #ax.set_xlabel('iterations')
    Energy=np.array(Energy);
    EnergyVariance=np.array(EnergyVariance);
    return IsingNQS, Energy, EnergyVariance
                


    
def TrainIsingAnnealedNQS(J,hx,hy,hz,N,alpha,eps,EpochNum,EpochSize,NumBurn,Skip,eta,reg):
    
    IsingNQS = NeuralQuantumState(N,alpha,eps)
    
    M = alpha*N
    Energy = []
    EnergyVariance = []
    
    for Epoch in range(EpochNum):
        
        PhysicalSamples, SampleThetas = IsingNQS.SampleDistribution(EpochSize,NumBurn,Skip)
        
        #PhysicalSamples = IsingNQS.generate_samples(EpochSize)[0]
        #SampleThetas = IsingNQS.ComputeAllThetas(PhysicalSamples)
                
        SPFx, SPFy = IsingNQS.SumPsiFracs(PhysicalSamples,SampleThetas)
        E = IsingLocalEnergy(N,PhysicalSamples,SPFx,SPFy,J,hx,hy,hz)
                
        aD, bD = IsingNQS.Gradients(PhysicalSamples,SampleThetas)
            
        MaD = np.mean(aD,axis=0)
        MbD = np.mean(bD,axis=0)
        #MWD = np.mean(WD,axis=0)
        MWD = 2*np.dot(aD.T,bD)/EpochSize
                
        #MEaD = np.mean(E[:,np.newaxis]*aD,axis=0)
        #MEbD = np.mean(E[:,np.newaxis]*bD,axis=0)
        #MEWD = np.mean(E[:,np.newaxis,np.newaxis]*WD,axis=0)
        
        MEaD = np.dot(E,aD)/EpochSize
        MEbD = np.dot(E,bD)/EpochSize
        MEWD = 2*np.dot(np.multiply(E,aD.T),bD)/EpochSize
        
        Eng = np.mean(E)
        Evar = np.var(E)
        Energy.append(Eng)
        EnergyVariance.append(Evar)
        
        Ga = MEaD - Eng*MaD
        Gb = MEbD - Eng*MbD
        GW = MEWD - Eng*MWD
        
        if Evar < 15:
            beta = eta*Evar/15
        else:
            beta = eta
        
        IsingNQS.GradientDescentStep(Ga,Gb,GW,beta,reg)
        
        print('%d. E=%.4f, E_loc = %.4f E_var=%.4f' % (Epoch+1,Eng,Eng/N,Evar))
    
    #ax = pd.Series(Energy).plot(title=r'Local energy $\frac{<E_{loc}>}{N}$',grid=True)
    #ax.set_ylabel(r'$<E_{loc}>/N$')
    #ax.set_xlabel('iterations')
    Energy=np.array(Energy);
    EnergyVariance=np.array(EnergyVariance);
    return IsingNQS, Energy, EnergyVariance    
    
    
    
def TrainIsingHigherEnergyNQS(J,hx,hy,hz,N,alpha,eps,EpochNum,EpochSize,NumBurn,Skip,eta,reg,lagrange):
    
    IsingGroundStateNQS, Energy0, EnergyVariance0 = TrainIsingNQS(J,hx,hy,hz,N,alpha,eps,EpochNum,EpochSize,NumBurn,Skip,eta,reg)
    
    IsingHigherStateNQS = NeuralQuantumState(N,alpha,eps)
    
    M = alpha*N
    Energy = []
    EnergyVariance = []
    Overlaps = []
    
    for Epoch in range(EpochNum):
        
        PhysicalSamples, SampleThetas = IsingHigherStateNQS.SampleDistribution(EpochSize,NumBurn,Skip)
        
        GroundThetas = IsingGroundStateNQS.ComputeAllThetas(PhysicalSamples)
        
        #PhysicalSamples = IsingNQS.generate_samples(EpochSize)[0]
        #SampleThetas = IsingNQS.ComputeAllThetas(PhysicalSamples)
                
        SPFx, SPFy = IsingHigherStateNQS.SumPsiFracs(PhysicalSamples,SampleThetas)
        E = IsingLocalEnergy(N,PhysicalSamples,SPFx,SPFy,J,hx,hy,hz)
        E = np.real(E)
        F = np.conj(IsingGroundStateNQS.PsiValue(PhysicalSamples,GroundThetas))/(IsingHigherStateNQS.PsiValue(PhysicalSamples,SampleThetas))
                
        aD, bD = IsingHigherStateNQS.Gradients(PhysicalSamples,SampleThetas)
            
        MaD = np.mean(aD,axis=0)
        MbD = np.mean(bD,axis=0)
        #MWD = np.mean(WD,axis=0)
        MWD = 2*np.dot(aD.T,bD)/EpochSize
                
        #MEaD = np.mean(E[:,np.newaxis]*aD,axis=0)
        #MEbD = np.mean(E[:,np.newaxis]*bD,axis=0)
        #MEWD = np.mean(E[:,np.newaxis,np.newaxis]*WD,axis=0)
        
        MEaD = np.dot(E,aD)/EpochSize
        MEbD = np.dot(E,bD)/EpochSize
        MEWD = 2*np.dot(np.multiply(E,aD.T),bD)/EpochSize
        
        Eng = np.mean(E)
        Evar = np.var(E)
        Energy.append(Eng)
        EnergyVariance.append(Evar)
        
        Fav = np.mean(F)
        Overlaps.append(Fav)
        
        MFaD = np.dot(F,aD)/EpochSize
        MFbD = np.dot(F,bD)/EpochSize
        MFWD = 2*np.dot(np.multiply(F,aD.T),bD)/EpochSize
        
        Ga = MEaD - Eng*MaD
        Gb = MEbD - Eng*MbD
        GW = MEWD - Eng*MWD
        
        La = np.real(np.conj(Fav)*(MFaD - 2*Fav*np.real(MaD)))
        Lb = np.real(np.conj(Fav)*(MFbD - 2*Fav*np.real(MbD)))
        LW = np.real(np.conj(Fav)*(MFWD - 2*Fav*np.real(MWD)))
        
        IsingHigherStateNQS.GradientDescentStep(Ga+lagrange*La,Gb+lagrange*Lb,GW+lagrange*LW,eta,reg)
        
        print('%d. E=%.4f, E_loc = %.4f E_var = %.4f, F_av = %.4f,%.4f' % (Epoch+1,Eng,Eng/N,Evar,np.real(Fav),np.imag(Fav)))
    
    #ax = pd.Series(Energy).plot(title=r'Local energy $\frac{<E_{loc}>}{N}$',grid=True)
    #ax.set_ylabel(r'$<E_{loc}>/N$')
    #ax.set_xlabel('iterations')
    Energy=np.array(Energy);
    EnergyVariance=np.array(EnergyVariance);
    return IsingHigherStateNQS, IsingGroundStateNQS, Energy, EnergyVariance, Overlaps    
    
    
    
    
    
    
    
def Train2DIsingNQS(J,hx,hy,hz,Na,Nb,alpha,eps,EpochNum,EpochSize,NumBurn,Skip,eta,reg):
 
    N = Na*Nb
    
    IsingNQS = NeuralQuantumState(N,alpha,eps)
    
    M = alpha*N
    Energy = []
    EnergyVariance = []
    
    for Epoch in range(EpochNum):
        
        PhysicalSamples, SampleThetas = IsingNQS.SampleDistribution(EpochSize,NumBurn,Skip)
        
        #PhysicalSamples = IsingNQS.generate_samples(EpochSize)[0]
        #SampleThetas = IsingNQS.ComputeAllThetas(PhysicalSamples)
                
        SPFx,SPFy = IsingNQS.SumPsiFracs(PhysicalSamples,SampleThetas)
        E = TwoDIsingLocalEnergy(N,Na,Nb,PhysicalSamples,SPFx,SPFy,J,hx,hy,hz,EpochSize)
                
        aD, bD = IsingNQS.Gradients(PhysicalSamples,SampleThetas)
            
        MaD = np.mean(aD,axis=0)
        MbD = np.mean(bD,axis=0)
        #MWD = np.mean(WD,axis=0)
        MWD = 2*np.dot(aD.T,bD)/EpochSize
                
        #MEaD = np.mean(E[:,np.newaxis]*aD,axis=0)
        #MEbD = np.mean(E[:,np.newaxis]*bD,axis=0)
        #MEWD = np.mean(E[:,np.newaxis,np.newaxis]*WD,axis=0)
        
        MEaD = np.dot(E,aD)/EpochSize
        MEbD = np.dot(E,bD)/EpochSize
        MEWD = 2*np.dot(np.multiply(E,aD.T),bD)/EpochSize
        
        Eng = np.mean(E)
        Evar = np.var(E)
        Energy.append(Eng)
        EnergyVariance.append(Evar)
        
        Ga = MEaD - Eng*MaD
        Gb = MEbD - Eng*MbD
        GW = MEWD - Eng*MWD
        
        IsingNQS.GradientDescentStep(Ga,Gb,GW,eta,reg)
        
        print('%d. E=%.4f, E_loc = %.4f E_var=%.4f' % (Epoch+1,Eng,Eng/N,Evar))
    

    Energy=np.array(Energy);
    EnergyVariance=np.array(EnergyVariance);
    return IsingNQS, Energy, EnergyVariance
    
    
def Train3DIsingNQS(J,hx,hy,hz,Na,Nb,Nc,alpha,eps,EpochNum,EpochSize,NumBurn,Skip,eta,reg):
 
    N = Na*Nb*Nc
    
    IsingNQS = NeuralQuantumState(N,alpha,eps)
    
    M = alpha*N
    Energy = []
    EnergyVariance = []
    
    for Epoch in range(EpochNum):
        
        PhysicalSamples, SampleThetas = IsingNQS.SampleDistribution(EpochSize,NumBurn,Skip)
        
        #PhysicalSamples = IsingNQS.generate_samples(EpochSize)[0]
        #SampleThetas = IsingNQS.ComputeAllThetas(PhysicalSamples)
                
        SPFx,SPFy = IsingNQS.SumPsiFracs(PhysicalSamples,SampleThetas)
        E = ThreeDIsingLocalEnergy(N,Na,Nb,Nc,PhysicalSamples,SPFx,SPFy,J,hx,hy,hz,EpochSize)
                
        aD, bD = IsingNQS.Gradients(PhysicalSamples,SampleThetas)
            
        MaD = np.mean(aD,axis=0)
        MbD = np.mean(bD,axis=0)
        #MWD = np.mean(WD,axis=0)
        MWD = 2*np.dot(aD.T,bD)/EpochSize
                
        #MEaD = np.mean(E[:,np.newaxis]*aD,axis=0)
        #MEbD = np.mean(E[:,np.newaxis]*bD,axis=0)
        #MEWD = np.mean(E[:,np.newaxis,np.newaxis]*WD,axis=0)
        
        MEaD = np.dot(E,aD)/EpochSize
        MEbD = np.dot(E,bD)/EpochSize
        MEWD = 2*np.dot(np.multiply(E,aD.T),bD)/EpochSize
        
        Eng = np.mean(E)
        Evar = np.var(E)
        Energy.append(Eng)
        EnergyVariance.append(Evar)
        
        Ga = MEaD - Eng*MaD
        Gb = MEbD - Eng*MbD
        GW = MEWD - Eng*MWD
        
        IsingNQS.GradientDescentStep(Ga,Gb,GW,eta,reg)
        
        print('%d. E=%.4f, E_loc = %.4f E_var=%.4f' % (Epoch+1,Eng,Eng/N,Evar))
    
    #ax = pd.Series(Energy).plot(title=r'Local energy $\frac{<E_{loc}>}{N}$',grid=True)
    #ax.set_ylabel(r'$<E_{loc}>/N$')
    #ax.set_xlabel('iterations')
    Energy=np.array(Energy);
    EnergyVariance=np.array(EnergyVariance);
    return IsingNQS, Energy, EnergyVariance    
    
    
def TrainHeisenbergNQS(Jx,Jy,Jz,hx,hy,hz,N,alpha,eps,EpochNum,EpochSize,NumBurn,Skip,eta,reg):
    
    IsingNQS = NeuralQuantumState(N,alpha,eps)
    
    M = alpha*N
    Energy = []
    EnergyVariance = []
    
    for Epoch in range(EpochNum):
        
        PhysicalSamples, SampleThetas = IsingNQS.SampleDistribution(EpochSize,NumBurn,Skip)
        
        #PhysicalSamples = IsingNQS.generate_samples(EpochSize)[0]
        #SampleThetas = IsingNQS.ComputeAllThetas(PhysicalSamples)
                
        SPFx,SPFy = IsingNQS.SumPsiFracs(PhysicalSamples,SampleThetas)
        SPFxx,SPFyy = IsingNQS.SumPsiFracs2(PhysicalSamples,SampleThetas,EpochSize)
        E = HeisenbergLocalEnergy(N,Jx,Jy,Jz,hx,hy,hz,SPFx,SPFy,SPFxx,SPFyy,PhysicalSamples)
                
        aD, bD = IsingNQS.Gradients(PhysicalSamples,SampleThetas)
            
        MaD = np.mean(aD,axis=0)
        MbD = np.mean(bD,axis=0)
        #MWD = np.mean(WD,axis=0)
        MWD = 2*np.dot(aD.T,bD)/EpochSize
                
        #MEaD = np.mean(E[:,np.newaxis]*aD,axis=0)
        #MEbD = np.mean(E[:,np.newaxis]*bD,axis=0)
        #MEWD = np.mean(E[:,np.newaxis,np.newaxis]*WD,axis=0)
        
        MEaD = np.dot(E,aD)/EpochSize
        MEbD = np.dot(E,bD)/EpochSize
        MEWD = 2*np.dot(np.multiply(E,aD.T),bD)/EpochSize
        
        Eng = np.mean(E)
        Evar = np.var(E)
        Energy.append(Eng)
        EnergyVariance.append(Evar)
        
        Ga = MEaD - Eng*MaD
        Gb = MEbD - Eng*MbD
        GW = MEWD - Eng*MWD
        
        IsingNQS.GradientDescentStep(Ga,Gb,GW,eta,reg)
        
        print('%d. E=%.4f, E_loc = %.4f E_var=%.4f' % (Epoch+1,Eng,Eng/N,Evar))
    
    #ax = pd.Series(Energy).plot(title=r'Local energy $\frac{<E_{loc}>}{N}$',grid=True)
    #ax.set_ylabel(r'$<E_{loc}>/N$')
    #ax.set_xlabel('iterations')
    Energy=np.array(Energy);
    EnergyVariance=np.array(EnergyVariance);
    return IsingNQS, Energy, EnergyVariance        
    
    
        
    
class NeuralQuantumState:
    
    
    def __init__(self,N,alpha,eps):
    
        self.a = (np.random.random(N) - 0.5)*eps
        M = N*alpha
        self.b = (np.random.random(M) - 0.5)*eps
        self.N = N
        self.M = M
        self.W = (np.random.random((N,M)) - 0.5)*eps
    
    
    
    def ComputeThetas(self,PhysicalState):
        
        return self.b + np.dot(self.W.T,PhysicalState)
    
    def ComputeAllThetas(self,PhysicalState):
        
        return self.b[np.newaxis,:] + np.dot(PhysicalState,self.W)    
    
    def ComputeGammas(self,HiddenState):
        
        return self.a + np.dot(self.W,HiddenState)
    
    
    
    def PsiValue(self,PhysicalState,thetas):
        
        return np.exp(np.sum(self.a[np.newaxis,:]*PhysicalState,axis=1))*np.prod(np.cosh(thetas),axis=1)
    
    def Wavefunc(self, PhysicalState):
        return self.PsiValue(PhysicalState, self.ComputeAllThetas(PhysicalState));
    
    def SumPsiFracs(self,PhysicalState,thetas):
    
        Part1 = np.exp(-2.0*self.a[np.newaxis,:]*PhysicalState)
        Part2 = np.prod(np.cosh(thetas[:,np.newaxis,:] - 2.0*PhysicalState[:,:,np.newaxis]*self.W[np.newaxis,:,:])/np.cosh(thetas[:,np.newaxis,:]),axis=2)
        
        return np.sum(Part1*Part2,axis=1), np.sum(1j*PhysicalState*Part1*Part2,axis=1)
    
    
    def SumPsiFracs2(self,PhysicalState,thetas,NS):
    
        Part1 = np.exp(-2.0*self.a[np.newaxis,:]*PhysicalState)
        Part11 = np.zeros((NS,self.N))
        Part11[:,0:(self.N-1)] = Part1[:,0:(self.N-1)]*Part1[:,1:self.N]
        Part11[:,(self.N-1)] = Part1[:,(self.N-1)]*Part1[:,0]
        
        Part2 = PhysicalState[:,:,np.newaxis]*self.W[np.newaxis,:,:]
        Part21 = np.zeros(np.shape(Part2))
        Part21[:,0:(self.N-1),:] = Part2[:,0:(self.N-1),:]*Part2[:,1:self.N,:]
        Part21[:,(self.N-1),:] = Part2[:,(self.N-1),:]*Part2[:,0,:]
        Part22 = np.prod(np.cosh(thetas[:,np.newaxis,:] - 2.0*Part21)/np.cosh(thetas[:,np.newaxis,:]),axis=2)
        
        NegativeScreen = np.zeros((NS,self.N))
        NegativeScreen[:,0:(self.N-1)] = PhysicalState[:,0:(self.N-1)]*PhysicalState[:,1:self.N]
        NegativeScreen[:,(self.N-1)] = PhysicalState[:,(self.N-1)]*PhysicalState[:,0]
        
        return np.sum(Part11*Part22,axis=1), np.sum(-1.0*NegativeScreen*Part11*Part22,axis=1)
    
    def Gradients(self,PhysicalSamples,SampleThetas):
    
        Tthetas = np.tanh(SampleThetas)
 
        return 0.5*PhysicalSamples, 0.5*Tthetas, #PhysicalSamples[:,:,np.newaxis]*Tthetas[:,np.newaxis,:] 
    
    def GradientDescentStep(self,da,db,dW,eta,reg):
        
        self.a = (1-reg)*self.a - eta*da
        self.b = (1-reg)*self.b - eta*db
        self.W = (1-reg)*self.W - eta*dW
        
    
    def AdamGradientStep(self,t,da,db,dW,eta,eps2,beta1,beta2,MaOld,MbOld,MWOld,VaOld,VbOld,VWOld):
        
        Ma = beta1*MaOld + (1-beta1)*da
        Va = beta2*VaOld + (1-beta2)*da2
        
        HMa = Ma/(1-beta1**(t+1))
        HVa = Va/(1-beta2**(t+1))
        
        Mb = beta1*MbOld + (1-beta1)*db
        Vb = beta2*VbOld + (1-beta2)*db2
        
        HMb = Mb/(1-beta1**(t+1))
        HVb = Vb/(1-beta2**(t+1))
    
        MW = beta1*MWOld + (1-beta1)*dW
        VW = beta2*VWOld + (1-beta2)*dW2
        
        HMW = MW/(1-beta1**(t+1))
        HVW = VW/(1-beta2**(t+1))
        
        self.a = self.a - eta*HMa/(np.sqrt(HVa)+eps2)
        self.b = self.b - eta*HMb/(np.sqrt(HVb)+eps2)
        self.W = self.W - eta*HMW/(np.sqrt(HVW)+eps2)
        
    def Logistic2(self,x):    
        
        return 1.0/(1+np.exp(-2*x))
    
    
    
    def SampleDistribution(self,NumSamples,NumBurn,Skip):
        
        PhysicalState = 2*(np.random.randint(2,size=self.N))-1
        HiddenState = 2*(np.random.randint(2,size=self.M))-1
        
        thetas = self.ComputeThetas(PhysicalState)
        gammas = self.ComputeGammas(HiddenState)
        
        PhysicalSamples = np.zeros((NumSamples,self.N))
        HiddenSamples = np.zeros((NumSamples,self.M))
        
        SampleThetas = np.zeros((int(NumSamples/Skip),int(self.M)))
        SampleGammas = np.zeros((int(NumSamples/Skip),int(self.N)))
        
        for burn in range(NumBurn):
            
            PhysicalState, HiddenState, thetas, gammas = self.NextSample(thetas,gammas)
            
        for sample in range(NumSamples):
            
            PhysicalState, HiddenState, thetas, gammas = self.NextSample(thetas,gammas)
            
            if (sample+1)%Skip == 0:
                PhysicalSamples[sample,:] = PhysicalState
                SampleThetas[sample,:] = thetas
            
        return PhysicalSamples, SampleThetas
    
    def NextSample(self, Oldthetas, Oldgammas):
        
        LogisticTheta = self.Logistic2(Oldthetas)
        LogisticGamma = self.Logistic2(Oldgammas)
        
        PhysicalState = np.ones(self.N)
        HiddenState = np.ones(self.M)
        
        l=np.random.uniform(size=(self.N))
        r=np.random.uniform(size=(self.M))
        
        HiddenState[r>LogisticTheta] = -1
        PhysicalState[l>LogisticGamma] = -1
        
        thetas = self.ComputeThetas(PhysicalState)
        gammas = self.ComputeGammas(HiddenState)
        
        return PhysicalState, HiddenState, thetas, gammas
    


    
    
    
    
    
    
    
    
    
    def Gammai(self,h): #analogue of effective_angles, but with same indices as spins
        return self.a+np.inner(self.W,h);

    def effective_angles(self,state): #correspond to same indices as hidden
        return self.b+np.inner(np.transpose(self.W),state)

    def Logistic(self,Gammai):
        return 1./(1+ np.exp(-2*Gammai));#logistic function is essentially sigmoid
    
    
    def generate_samples(self,NS):
        state_list=np.zeros((NS,self.N));
        h_list=np.zeros((NS,self.M));

    
        currenth = np.random.randint(2, size=self.M);#generate random initial hidden units
        currentstate = np.random.randint(2, size=self.N)# generate random initial spin state
        
        state_i = list(range(self.N))
        h_i=list(range(self.M))

    
        for sample in range(NS):
        
            R=np.random.uniform(size=(self.N));
            hR=np.random.uniform(size=(self.M));
            newstate=currentstate;
            newh=currenth;
            Ps1h=self.Logistic(self.Gammai(currenth));#P(sigma=1|h)
            Ph1s=self.Logistic(self.effective_angles(currentstate));#P(h=1|sigma))
            for i in state_i:

                if Ps1h[i]>R[i]:
                    newstate[i]=1;
                else:
                    newstate[i]=-1;

            for j in h_i:
                if Ph1s[j]>hR[j]:
                    newh[j]=1;
                else:
                    newh[j]=-1;


            state_list[sample]=newstate;
            h_list[sample]=newh;
        
            currentstate=newstate;
            currenth=newh;
        
        return (state_list, h_list);


#fm=generate_samples(10);
#print("new states",fm[0]);
#print("new h",fm[1]);
    