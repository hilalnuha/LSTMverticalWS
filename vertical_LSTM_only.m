clc
clear all
close all
%M=csvread("WS_10m.csv",1,3);
%M=csvread("WS_hr.csv",2,3,[2 3 3000 3]);
M=csvread("WS_KFUPM_10m_2015.csv",1,2);
% [10 20 30 40 60 80 100 120 140 160 180]
% [10 20 30 40 x50 60 x70 80 x90 100 x110 120 x130 140 x150 160 x170 180]
%M=M(:,[2 4 5 6 7 8 10 

days=360;
%numdat=6*24*days;
inputsize=4;
%M=M(1:(numdat),:);
N=length(M);

for k=1:11
for i=1:N  
    if M(i,k)> 20 %CLEAN THE DATA if 9999, then replace with previous value
        M(i,k)=M(i-1,k);
    elseif M(i,k)<= 0
    M(i,k)=M(i-1,k);
    end
end
end

M=fliplr(M);

ii=1;
for i=1:N
    diff0=M(i,2:11)-M(i,1:10);
    lt0=sum(find(diff0<0.1));
    if lt0==0 
        MN(ii,:)=M(i,:);
        ii=ii+1;
    end
end

mt15=find(MN(:,6)<=15);   
M=MN(mt15,:);
mt10=find(M(:,3)<=10);   
M=M(mt10,:);

M=[M(:,[1 2 3 4]) (M(:,4)+M(:,5))/2 M(:,5) (M(:,5)+M(:,6))/2 M(:,6) (M(:,6)+M(:,7))/2 M(:,7) (M(:,7)+M(:,8))/2 M(:,8) (M(:,8)+M(:,9))/2 M(:,9) (M(:,9)+M(:,10))/2 M(:,10) (M(:,10)+M(:,11))/2 M(:,11)];
% [10 20 30 40 60 80 100 120 140 160 180]
% [10 20 30 40 x50 60 x70 80 x90 100 x110 120 x130 140 x150 160 x170 180]
%


numdat=length(M);

%R=6; % Every 6 makes an hour
%mm=floor(N/R);
%for i=1:mm
%    j=(i-1)*R+1;
%    MD(i,1)=mean(M(j:j+R-1));
%end

trainingnum=floor(0.8*numdat); % Num of training samples
maxx=max(max(M(1:trainingnum,1:inputsize)));
training=M(1:trainingnum,:);

series=training/maxx;
datasize=size(series);
nex=1;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testing=M((trainingnum+1):end,:);

seriesT=testing/maxx;
%numdata=max(datasize)-(inputsize+ahead-1);
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P50 = traininginput';
Y50 = trainingtarget';
Ptest50 = testinginput';
Ytest50 = testingtarget';
testingtarget50=Ytest50'*maxx;
%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P50,Y50,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P50,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf50train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest50,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf50=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest50=mse(RNNOutf50,testingtarget50);
mapetest50=mape(RNNOutf50,testingtarget50);
mbetest50=mbe(RNNOutf50,testingtarget50);
r2test50=rsquare(RNNOutf50,testingtarget50);
RNNperf50=[msetest50 mapetest50 mbetest50 r2test50];
target50=testingtarget50;

%
figure
PtestMax50=Ptest50'*maxx;
height50=[10 20 30 40 50];
plot([PtestMax50'; RNNOutf50' ],height50);
rang50=[0 13];
xlim(rang50)
%ylim([-0.4 0.8])
title(['RNN 50 m Testing MSE=' num2str(msetest50) ',MAPE=' num2str(mapetest50) ',MBE=' num2str(mbetest50) ',R^2=' num2str(r2test50)]);
%
figure
rl50=[1:13];
plot( RNNOutf50, target50,'ob',rl50,rl50,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl50)+0.5]);
ylim([0 max(rl50)+0.5]);
title(['RNN 50 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test50*100,2)) ' %'])

%xlim([0 13])
%ylim([-0.4 0.8])

%camroll(90)

%
MLPP50 = traininginput';
MLPY50 = trainingtarget';
MLPPtest50 = testinginput';
MLPYtest50 = testingtarget';
MLPtestingtarget50=MLPYtest50'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP50)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
netMLP.trainParam.showWindow = 0; 
[netMLP,tr,Y,E] = train(netMLP,MLPP50,MLPY50);

outval = netMLP(MLPP50);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest50);
outvaltestmax=outvaltest'*maxx;
MLPOutf50=outvaltestmax;
MLPtestingtargetmax50=testingtarget50;
MLPmsetest50=mse(MLPOutf50,MLPtestingtargetmax50);
MLPmapetest50=mape(MLPOutf50,MLPtestingtargetmax50);
MLPmbetest50=mbe(MLPOutf50,MLPtestingtargetmax50);
MLPr2test50=rsquare(MLPOutf50,MLPtestingtargetmax50);
MLPperf50=[MLPmsetest50 MLPmapetest50 MLPmbetest50 MLPr2test50];
MLPOutf50train=outvalmax;
MLPPtestMax50=MLPPtest50'*maxx;
%
figure

plot([MLPPtestMax50'; MLPOutf50' ],height50);
xlim(rang50)
%ylim([-0.4 0.8])
title(['MLP 50m Testing MSE=' num2str(MLPmsetest50) ',MAPE=' num2str(MLPmapetest50) ',MBE=' num2str(MLPmbetest50)  ',R^2=' num2str(MLPr2test50)]);
%camroll(90)
% [10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180]
%
figure
%rl=[1:13];
plot( MLPOutf50, testingtargetmax,'ob',rl50,rl50,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl50)+0.5]);
ylim([0 max(rl50)+0.5]);
title(['MLP 50 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test50*100,2)) ' %'])
% LSTM

LSTMP50 = traininginput';
LSTMY50 = trainingtarget';
LSTMPtest50 = testinginput';
LSTMYtest50 = testingtarget';
LSTMtestingtarget50=LSTMYtest50'*maxx;

%Create NN

numiter=20;
numhid=20;
miniBatchSize = 20;
numFeatures=inputsize+(nex-1);

numResponses = 1;
numHiddenUnits = numhid;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(10)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 15;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(LSTMP50,LSTMY50,layers,options);
outval = predict(net,LSTMP50,'MiniBatchSize',1);

outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvaltest = predict(net,LSTMPtest50,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
LSTMOutf50=outvaltestmax;
LSTMtestingtargetmax50=testingtarget50;
LSTMmsetest50=mse(LSTMOutf50,LSTMtestingtargetmax50);
LSTMmapetest50=mape(LSTMOutf50,LSTMtestingtargetmax50);
LSTMmbetest50=mbe(LSTMOutf50,LSTMtestingtargetmax50);
LSTMr2test50=rsquare(LSTMOutf50,LSTMtestingtargetmax50);
LSTMperf50=[LSTMmsetest50 LSTMmapetest50 LSTMmbetest50 LSTMr2test50];
LSTMOutf50train=outvalmax;
LSTMPtestMax50=LSTMPtest50'*maxx;

xlim(rang50)

figure

plot([LSTMPtestMax50'; LSTMOutf50' ],height50);
%ylim([-0.4 0.8])
title(['LSTM 50m Testing MSE=' num2str(LSTMmsetest50) ',MAPE=' num2str(LSTMmapetest50) ',MBE=' num2str(LSTMmbetest50)  ',R^2=' num2str(LSTMr2test50)]);
%camroll(90)
% [10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180]
%
figure
%rl=[1:13];
plot( LSTMOutf50, testingtargetmax,'ob',rl50,rl50,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl50)+0.5]);
ylim([0 max(rl50)+0.5]);
title(['LSTM 50 m Testing ']);
text(2,12,['R^2=' num2str(round(LSTMr2test50*100,2)) ' %'])

CNNP50 = traininginput';
[panj,leb]=size(CNNP50);
CNNP50CNN =reshape(CNNP50 ,[panj,1,1,leb]);

CNNY50 = trainingtarget';
CNNPtest50 = testinginput';
[panj,leb]=size(CNNPtest50);
CNNPtest50CNN =reshape(CNNPtest50 ,[panj,1,1,leb]);

CNNYtest50 = testingtarget';
CNNtestingtarget50=CNNYtest50'*maxx;

%Create NN

numiter=20;
numhid=20;
miniBatchSize = 20;
numFeatures=inputsize+(nex-1);

numResponses = 1;
numHiddenUnits = numhid;

layers = [ ...
    imageInputLayer([(inputsize+nex-1) 1 1])
    convolution2dLayer([inputsize+nex-2 1],inputsize+nex-2,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(1,'Stride',1) 
    fullyConnectedLayer(1)
    regressionLayer];

maxEpochs = 15;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(CNNP50CNN,CNNY50',layers,options);
outval = predict(net,CNNP50CNN,'MiniBatchSize',1);

outvalmax=outval*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvaltest = predict(net,CNNPtest50CNN,'MiniBatchSize',1);
outvaltestmax=outvaltest*maxx;
CNNOutf50=outvaltestmax;
CNNtestingtargetmax50=testingtarget50;
CNNmsetest50=mse(CNNOutf50,CNNtestingtargetmax50);
CNNmapetest50=mape(CNNOutf50,CNNtestingtargetmax50);
CNNmbetest50=mbe(CNNOutf50,CNNtestingtargetmax50);
CNNr2test50=rsquare(CNNOutf50,CNNtestingtargetmax50);
CNNperf50=[CNNmsetest50 CNNmapetest50 CNNmbetest50 CNNr2test50];
CNNOutf50train=outvalmax;
CNNPtestMax50=CNNPtest50'*maxx;

figure

plot([CNNPtestMax50'; CNNOutf50' ],height50);
xlim(rang50)
%ylim([-0.4 0.8])
title(['CNN 50m Testing MSE=' num2str(CNNmsetest50) ',MAPE=' num2str(CNNmapetest50) ',MBE=' num2str(CNNmbetest50)  ',R^2=' num2str(CNNr2test50)]);
%camroll(90)
% [10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180]
%
figure
%rl=[1:13];
plot( CNNOutf50, testingtargetmax,'ob',rl50,rl50,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl50)+0.5]);
ylim([0 max(rl50)+0.5]);
title(['CNN 50 m Testing ']);
text(2,12,['R^2=' num2str(round(CNNr2test50*100,2)) ' %'])


% 1/7 WSE

LWSEP50 = traininginput';
LWSEY50 = trainingtarget';
LWSEPtest50 = testinginput';
LWSEYtest50 = testingtarget';
LWSEtestingtarget50=LWSEYtest50'*maxx;


alpha=1/3;
outval=LWSEP50(4,:)*((50/40)^(alpha));
outvalmax=outval*maxx;
LWSEOutf50train=outvalmax;


outvaltest=MLPPtest50(4,:)*((50/40)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget50;
LWSEOutf50=outvaltestmax;
LWSEmsetest50=mse(LWSEOutf50,testingtarget50);
LWSEmapetest50=mape(LWSEOutf50,testingtarget50);
LWSEmbetest50=mbe(LWSEOutf50,testingtarget50);
LWSEr2test50=rsquare(LWSEOutf50,testingtarget50);
LWSEperf50=[LWSEmsetest50 LWSEmapetest50 LWSEmbetest50 LWSEr2test50];
LWSEPtestMax50=LWSEPtest50'*maxx;

figure
interv=1:200;
plot( [testingtarget50(interv) ],'b');
hold on
plot( [RNNOutf50(interv)],'r');
plot( [MLPOutf50(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

figure
plot(mean([seriesT(:,1:inputsize)'*maxx; testingtarget50' ]'),height50,'k');
hold on
plot(mean([PtestMax50'; RNNOutf50' ]'),height50,'b-s');
plot(mean([MLPPtestMax50'; MLPOutf50' ]'),height50,'--r');
plot(mean([LSTMPtestMax50'; LSTMOutf50' ]'),height50,'x-m');
plot(mean([CNNPtestMax50'; CNNOutf50' ]'),height50,'o-y');
plot(mean([LWSEPtestMax50'; LWSEOutf50' ]'),height50,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','LSTM','CNN','1/7 est','Location','northwest')

display ('50 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf50]
[MLPperf50]
[LSTMperf50]
[CNNperf50]
[LWSEperf50]

%% 60

nex=2;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P60 = [P50; RNNOutf50train'/maxx];
Y60 = trainingtarget';
Ptest60 = [Ptest50; RNNOutf50'/maxx];
Ytest60 = testingtarget';
testingtarget60=Ytest60'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P60,Y60,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P60,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf60train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest60,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf60=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest60=mse(RNNOutf60,testingtarget60);
mapetest60=mape(RNNOutf60,testingtarget60);
mbetest60=mbe(RNNOutf60,testingtarget60);
r2test60=rsquare(RNNOutf60,testingtarget60);
RNNperf60=[msetest60 mapetest60 mbetest60 r2test60];
target60=testingtarget60;

%
figure
PtestMax60=Ptest60'*maxx;
height60=[height50 60];
plot([PtestMax60'; RNNOutf60' ],height60);
rang60=[0 13.5];
xlim(rang60)
%ylim([-0.4 0.8])
title(['RNN 60 m Testing MSE=' num2str(msetest60) ',MAPE=' num2str(mapetest60) ',MBE=' num2str(mbetest60) ',R^2=' num2str(r2test60)]);
%
figure
rl60=[1:13.5];
plot( RNNOutf60, target60,'ob',rl60,rl60,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl60)+0.5]);
ylim([0 max(rl60)+0.5]);
title(['RNN 60 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test60*100,2)) ' %'])

%xlim([0 13])
%ylim([-0.4 0.8])

%camroll(90)
%
MLPP60= [MLPP50; MLPOutf50train'/maxx];
MLPY60 = trainingtarget';
MLPPtest60 = [MLPPtest50; MLPOutf50'/maxx];
MLPYtest60 = testingtarget';
MLPtestingtarget60=MLPYtest60'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP60)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP60,MLPY60);

outval = netMLP(MLPP60);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest60);
outvaltestmax=outvaltest'*maxx;
MLPOutf60=outvaltestmax;
MLPtestingtargetmax60=testingtarget60;
MLPmsetest60=mse(MLPOutf60,MLPtestingtargetmax60);
MLPmapetest60=mape(MLPOutf60,MLPtestingtargetmax60);
MLPmbetest60=mbe(MLPOutf60,MLPtestingtargetmax60);
MLPr2test60=rsquare(MLPOutf60,MLPtestingtargetmax60);
MLPperf60=[MLPmsetest60 MLPmapetest60 MLPmbetest60 MLPr2test60];
MLPOutf60train=outvalmax;
%
figure
MLPPtestMax60=MLPPtest60'*maxx;

plot([MLPPtestMax60'; MLPOutf60' ],height60);
xlim(rang60)
%ylim([-0.4 0.8])
title(['MLP 60m Testing MSE=' num2str(MLPmsetest60) ',MAPE=' num2str(MLPmapetest60) ',MBE=' num2str(MLPmbetest60)  ',R^2=' num2str(MLPr2test60)]);
%camroll(90)
% [10 20 30 40 60 60 70 80 90 100 110 120 130 140 160 160 170 180]
%
figure
%rl=[1:13];
plot( MLPOutf60, testingtargetmax,'ob',rl60,rl60,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl60)+0.5]);
ylim([0 max(rl60)+0.5]);
title(['MLP 60 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test60*100,2)) ' %'])

% LSTM

LSTMP60 = [LSTMP50; LSTMOutf50train'/maxx];
LSTMY60 = trainingtarget';
LSTMPtest60 = [LSTMPtest50; LSTMOutf50'/maxx];
LSTMYtest60 = testingtarget';
LSTMtestingtarget60=LSTMYtest60'*maxx;

%Create NN

numiter=20;
numhid=20;
miniBatchSize = 20;
numFeatures=inputsize+(nex-1);

numResponses = 1;
numHiddenUnits = numhid;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(10)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 15;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(LSTMP60,LSTMY60,layers,options);
outval = predict(net,LSTMP60,'MiniBatchSize',1);

outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvaltest = predict(net,LSTMPtest60,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
LSTMOutf60=outvaltestmax;
LSTMtestingtargetmax60=testingtarget60;
LSTMmsetest60=mse(LSTMOutf60,LSTMtestingtargetmax60);
LSTMmapetest60=mape(LSTMOutf60,LSTMtestingtargetmax60);
LSTMmbetest60=mbe(LSTMOutf60,LSTMtestingtargetmax60);
LSTMr2test60=rsquare(LSTMOutf60,LSTMtestingtargetmax60);
LSTMperf60=[LSTMmsetest60 LSTMmapetest60 LSTMmbetest60 LSTMr2test60];
LSTMOutf60train=outvalmax;
LSTMPtestMax60=LSTMPtest60'*maxx;

figure

plot([LSTMPtestMax60'; LSTMOutf60' ],height60);
xlim(rang60)
%ylim([-0.4 0.8])
title(['LSTM 60m Testing MSE=' num2str(LSTMmsetest60) ',MAPE=' num2str(LSTMmapetest60) ',MBE=' num2str(LSTMmbetest60)  ',R^2=' num2str(LSTMr2test60)]);
%camroll(90)
% [10 20 30 40 60 60 70 80 90 100 110 120 130 140 160 160 170 180]
%
figure
%rl=[1:13];
plot( LSTMOutf60, testingtargetmax,'ob',rl60,rl60,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl60)+0.5]);
ylim([0 max(rl60)+0.5]);
title(['LSTM 60 m Testing ']);
text(2,12,['R^2=' num2str(round(LSTMr2test60*100,2)) ' %'])


CNNP60 = [CNNP50; CNNOutf50train'/maxx];
[panj,leb]=size(CNNP60);
CNNP60CNN =reshape(CNNP60 ,[panj,1,1,leb]);

CNNY60 =  trainingtarget';
CNNPtest60 = [CNNPtest50; CNNOutf50'/maxx];
[panj,leb]=size(CNNPtest60);
CNNPtest60CNN =reshape(CNNPtest60 ,[panj,1,1,leb]);

CNNYtest60 = testingtarget';
CNNtestingtarget60=CNNYtest60'*maxx;

%Create NN

numiter=20;
numhid=20;
miniBatchSize = 20;
numFeatures=inputsize+(nex-1);

numResponses = 1;
numHiddenUnits = numhid;

layers = [ ...
    imageInputLayer([(inputsize+nex-1) 1 1])
    convolution2dLayer([inputsize+nex-2 1],inputsize+nex-2,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(1,'Stride',1) 
    fullyConnectedLayer(1)
    regressionLayer];

maxEpochs = 15;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(CNNP60CNN,CNNY60',layers,options);
outval = predict(net,CNNP60CNN,'MiniBatchSize',1);

outvalmax=outval*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvaltest = predict(net,CNNPtest60CNN,'MiniBatchSize',1);
outvaltestmax=outvaltest*maxx;
CNNOutf60=outvaltestmax;
CNNtestingtargetmax60=testingtarget60;
CNNmsetest60=mse(CNNOutf60,CNNtestingtargetmax60);
CNNmapetest60=mape(CNNOutf60,CNNtestingtargetmax60);
CNNmbetest60=mbe(CNNOutf60,CNNtestingtargetmax60);
CNNr2test60=rsquare(CNNOutf60,CNNtestingtargetmax60);
CNNperf60=[CNNmsetest60 CNNmapetest60 CNNmbetest60 CNNr2test60];
CNNOutf60train=outvalmax;
CNNPtestMax60=CNNPtest60'*maxx;

figure

plot([CNNPtestMax60'; CNNOutf60' ],height60);
xlim(rang60)
%ylim([-0.4 0.8])
title(['CNN 60m Testing MSE=' num2str(CNNmsetest60) ',MAPE=' num2str(CNNmapetest60) ',MBE=' num2str(CNNmbetest60)  ',R^2=' num2str(CNNr2test60)]);
%camroll(90)
% [10 20 30 40 60 60 70 80 90 100 110 120 130 140 160 160 170 180]
%
figure
%rl=[1:13];
plot( CNNOutf60, testingtargetmax,'ob',rl60,rl60,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl60)+0.5]);
ylim([0 max(rl60)+0.5]);
title(['CNN 60 m Testing ']);
text(2,12,['R^2=' num2str(round(CNNr2test60*100,2)) ' %'])


% 1/7 WSE
%
LWSEP60 = [LWSEP50; LWSEOutf50train/maxx];
LWSEY60 = trainingtarget';
LWSEPtest60 = [LWSEPtest50; LWSEOutf50'/maxx];
LWSEYtest60 = testingtarget';
LWSEtestingtarget60=LWSEYtest60'*maxx;


alpha=1/3;
outval=LWSEP60(5,:)*((60/50)^(alpha));
outvalmax=outval*maxx;
LWSEOutf60train=outvalmax;


outvaltest=MLPPtest60(5,:)*((60/50)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget60;
LWSEOutf60=outvaltestmax;
LWSEmsetest60=mse(LWSEOutf60,testingtarget60);
LWSEmapetest60=mape(LWSEOutf60,testingtarget60);
LWSEmbetest60=mbe(LWSEOutf60,testingtarget60);
LWSEr2test60=rsquare(LWSEOutf60,testingtarget60);
LWSEperf60=[LWSEmsetest60 LWSEmapetest60 LWSEmbetest60 LWSEr2test60];
LWSEPtestMax60=LWSEPtest60'*maxx;

figure
interv=1:200;
plot( [testingtarget60(interv) ],'b');
hold on
plot( [RNNOutf60(interv)],'r');
plot( [MLPOutf60(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

figure
plot(mean([seriesT(:,1:inputsize+(nex-1))'*maxx; testingtarget60' ]'),height60,'k');
hold on
plot(mean([PtestMax60'; RNNOutf60' ]'),height60,'b-s');
plot(mean([MLPPtestMax60'; MLPOutf60' ]'),height60,'--r');
plot(mean([LSTMPtestMax60'; LSTMOutf60' ]'),height60,'x-m');
plot(mean([CNNPtestMax60'; CNNOutf60' ]'),height60,'o-y');
plot(mean([LWSEPtestMax60'; LWSEOutf60' ]'),height60,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','LSTM','CNN','1/7 est','Location','northwest')

display ('60 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf60]
[MLPperf60]
[LSTMperf60]
[CNNperf60]
[LWSEperf60]

%% 70

nex=3;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P70 = [P60; RNNOutf60train'/maxx];
Y70 = trainingtarget';
Ptest70 = [Ptest60; RNNOutf60'/maxx];
Ytest70 = testingtarget';
testingtarget70=Ytest70'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P70,Y70,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P70,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf70train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest70,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf70=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest70=mse(RNNOutf70,testingtarget70);
mapetest70=mape(RNNOutf70,testingtarget70);
mbetest70=mbe(RNNOutf70,testingtarget70);
r2test70=rsquare(RNNOutf70,testingtarget70);
RNNperf70=[msetest70 mapetest70 mbetest70 r2test70];
target70=testingtarget70;

%
figure
PtestMax70=Ptest70'*maxx;
height70=[height60 70];
plot([PtestMax70'; RNNOutf70' ],height70);
rang70=[0 14];
xlim(rang70)
%ylim([-0.4 0.8])
title(['RNN 70 m Testing MSE=' num2str(msetest70) ',MAPE=' num2str(mapetest70) ',MBE=' num2str(mbetest70) ',R^2=' num2str(r2test70)]);
%
figure
rl70=[1:14];
plot( RNNOutf70, target70,'ob',rl70,rl70,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl70)+0.5]);
ylim([0 max(rl70)+0.5]);
title(['RNN 70 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test70*100,2)) ' %'])

%xlim([0 13])
%ylim([-0.4 0.8])

%camroll(90)
%
MLPP70= [MLPP60; MLPOutf60train'/maxx];
MLPY70 = trainingtarget';
MLPPtest70 = [MLPPtest60; MLPOutf60'/maxx];
MLPYtest70 = testingtarget';
MLPtestingtarget70=MLPYtest70'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP70)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP70,MLPY70);

outval = netMLP(MLPP70);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest70);
outvaltestmax=outvaltest'*maxx;
MLPOutf70=outvaltestmax;
MLPtestingtargetmax70=testingtarget70;
MLPmsetest70=mse(MLPOutf70,MLPtestingtargetmax70);
MLPmapetest70=mape(MLPOutf70,MLPtestingtargetmax70);
MLPmbetest70=mbe(MLPOutf70,MLPtestingtargetmax70);
MLPr2test70=rsquare(MLPOutf70,MLPtestingtargetmax70);
MLPperf70=[MLPmsetest70 MLPmapetest70 MLPmbetest70 MLPr2test70];
MLPOutf70train=outvalmax;
%
figure
MLPPtestMax70=MLPPtest70'*maxx;

plot([MLPPtestMax70'; MLPOutf70' ],height70);
xlim(rang70)
%ylim([-0.4 0.8])
title(['MLP 70m Testing MSE=' num2str(MLPmsetest70) ',MAPE=' num2str(MLPmapetest70) ',MBE=' num2str(MLPmbetest70)  ',R^2=' num2str(MLPr2test70)]);
%camroll(90)
% [10 20 30 40 70 70 70 80 90 100 110 120 130 140 170 170 170 180]
%
figure
%rl=[1:13];
plot( MLPOutf70, testingtargetmax,'ob',rl70,rl70,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl70)+0.5]);
ylim([0 max(rl70)+0.5]);
title(['MLP 70 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test70*100,2)) ' %'])

% LSTM

LSTMP70 = [LSTMP60; LSTMOutf60train'/maxx];
LSTMY70 = trainingtarget';
LSTMPtest70 = [LSTMPtest60; LSTMOutf60'/maxx];
LSTMYtest70 = testingtarget';
LSTMtestingtarget70=LSTMYtest70'*maxx;

%Create NN

numiter=20;
numhid=20;
miniBatchSize = 20;
numFeatures=inputsize+(nex-1);

numResponses = 1;
numHiddenUnits = numhid;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(10)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 15;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(LSTMP70,LSTMY70,layers,options);
outval = predict(net,LSTMP70,'MiniBatchSize',1);

outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvaltest = predict(net,LSTMPtest70,'MiniBatchSize',1);
outvaltestmax=outvaltest'*maxx;
LSTMOutf70=outvaltestmax;
LSTMtestingtargetmax70=testingtarget70;
LSTMmsetest70=mse(LSTMOutf70,LSTMtestingtargetmax70);
LSTMmapetest70=mape(LSTMOutf70,LSTMtestingtargetmax70);
LSTMmbetest70=mbe(LSTMOutf70,LSTMtestingtargetmax70);
LSTMr2test70=rsquare(LSTMOutf70,LSTMtestingtargetmax70);
LSTMperf70=[LSTMmsetest70 LSTMmapetest70 LSTMmbetest70 LSTMr2test70];
LSTMOutf70train=outvalmax;
LSTMPtestMax70=LSTMPtest70'*maxx;

figure

plot([LSTMPtestMax70'; LSTMOutf70' ],height70);
xlim(rang70)
%ylim([-0.4 0.8])
title(['LSTM 70m Testing MSE=' num2str(LSTMmsetest70) ',MAPE=' num2str(LSTMmapetest70) ',MBE=' num2str(LSTMmbetest70)  ',R^2=' num2str(LSTMr2test70)]);
%camroll(90)
% [10 20 30 40 70 70 70 80 90 100 110 120 130 140 170 170 170 180]
%
figure
%rl=[1:13];
plot( LSTMOutf70, testingtargetmax,'ob',rl70,rl70,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl70)+0.5]);
ylim([0 max(rl70)+0.5]);
title(['LSTM 70 m Testing ']);
text(2,12,['R^2=' num2str(round(LSTMr2test70*100,2)) ' %'])


CNNP70 = [CNNP60; CNNOutf60train'/maxx];
[panj,leb]=size(CNNP70);
CNNP70CNN =reshape(CNNP70 ,[panj,1,1,leb]);

CNNY70 =  trainingtarget';
CNNPtest70 = [CNNPtest60; CNNOutf60'/maxx];
[panj,leb]=size(CNNPtest70);
CNNPtest70CNN =reshape(CNNPtest70 ,[panj,1,1,leb]);

CNNYtest70 = testingtarget';
CNNtestingtarget70=CNNYtest70'*maxx;

%Create NN

numiter=20;
numhid=20;
miniBatchSize = 20;
numFeatures=inputsize+(nex-1);

numResponses = 1;
numHiddenUnits = numhid;

layers = [ ...
    imageInputLayer([(inputsize+nex-1) 1 1])
    convolution2dLayer([inputsize+nex-2 1],inputsize+nex-2,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(1,'Stride',1) 
    fullyConnectedLayer(1)
    regressionLayer];

maxEpochs = 15;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','none',...
    'Verbose',0);

net = trainNetwork(CNNP70CNN,CNNY70',layers,options);
outval = predict(net,CNNP70CNN,'MiniBatchSize',1);

outvalmax=outval*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);
%
outvaltest = predict(net,CNNPtest70CNN,'MiniBatchSize',1);
outvaltestmax=outvaltest*maxx;
CNNOutf70=outvaltestmax;
CNNtestingtargetmax70=testingtarget70;
CNNmsetest70=mse(CNNOutf70,CNNtestingtargetmax70);
CNNmapetest70=mape(CNNOutf70,CNNtestingtargetmax70);
CNNmbetest70=mbe(CNNOutf70,CNNtestingtargetmax70);
CNNr2test70=rsquare(CNNOutf70,CNNtestingtargetmax70);
CNNperf70=[CNNmsetest70 CNNmapetest70 CNNmbetest70 CNNr2test70];
CNNOutf70train=outvalmax;
CNNPtestMax70=CNNPtest70'*maxx;

figure

plot([CNNPtestMax70'; CNNOutf70' ],height70);
xlim(rang70)
%ylim([-0.4 0.8])
title(['CNN 70m Testing MSE=' num2str(CNNmsetest70) ',MAPE=' num2str(CNNmapetest70) ',MBE=' num2str(CNNmbetest70)  ',R^2=' num2str(CNNr2test70)]);
%camroll(90)
% [10 20 30 40 70 70 70 80 90 100 110 120 130 140 170 170 170 180]
%
figure
%rl=[1:13];
plot( CNNOutf70, testingtargetmax,'ob',rl70,rl70,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl70)+0.5]);
ylim([0 max(rl70)+0.5]);
title(['CNN 70 m Testing ']);
text(2,12,['R^2=' num2str(round(CNNr2test70*100,2)) ' %'])


% 1/7 WSE
%
LWSEP70 = [LWSEP60; LWSEOutf60train/maxx];
LWSEY70 = trainingtarget';
LWSEPtest70 = [LWSEPtest60; LWSEOutf60'/maxx];
LWSEYtest70 = testingtarget';
LWSEtestingtarget70=LWSEYtest70'*maxx;


alpha=1/3;
outval=LWSEP70(6,:)*((70/60)^(alpha));
outvalmax=outval*maxx;
LWSEOutf70train=outvalmax;


outvaltest=MLPPtest70(6,:)*((70/60)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget70;
LWSEOutf70=outvaltestmax;
LWSEmsetest70=mse(LWSEOutf70,testingtarget70);
LWSEmapetest70=mape(LWSEOutf70,testingtarget70);
LWSEmbetest70=mbe(LWSEOutf70,testingtarget70);
LWSEr2test70=rsquare(LWSEOutf70,testingtarget70);
LWSEperf70=[LWSEmsetest70 LWSEmapetest70 LWSEmbetest70 LWSEr2test70];
LWSEPtestMax70=LWSEPtest70'*maxx;

figure
interv=1:200;
plot( [testingtarget70(interv) ],'b');
hold on
plot( [RNNOutf70(interv)],'r');
plot( [MLPOutf70(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

figure
plot(mean([seriesT(:,1:inputsize+(nex-1))'*maxx; testingtarget70' ]'),height70,'k');
hold on
plot(mean([PtestMax70'; RNNOutf70' ]'),height70,'b-s');
plot(mean([MLPPtestMax70'; MLPOutf70' ]'),height70,'--r');
plot(mean([LSTMPtestMax70'; LSTMOutf70' ]'),height70,'x-m');
plot(mean([CNNPtestMax70'; CNNOutf70' ]'),height70,'o-y');
plot(mean([LWSEPtestMax70'; LWSEOutf70' ]'),height70,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','LSTM','CNN','1/7 est','Location','northwest')

display ('70 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf70]
[MLPperf70]
[LSTMperf70]
[CNNperf70]
[LWSEperf70]
