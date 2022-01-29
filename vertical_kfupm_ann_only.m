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


perc=100;
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

%camroll(90)

%
ANNP50 = traininginput';
ANNY50 = trainingtarget';
ANNPtest50 = testinginput';
ANNYtest50 = testingtarget';
ANNtestingtarget50=ANNYtest50'*maxx;

%Create NN

numiter=1200;
numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;

netANN.divideParam.trainInd=1:max(size(ANNP50)); %training_index
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
netANN.trainParam.showWindow = 0; 
netANN.trainFcn = 'traingda';
%netANN.layers{1}.transferFcn = 'logsig';
%netANN.layers{2}.transferFcn = 'tansig';
netANN.trainParam.epochs=numiter;
[netANN,tr,Y,E] = train(netANN,ANNP50,ANNY50);

outval = netANN(ANNP50);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest50);
outvaltestmax=outvaltest'*maxx;
ANNOutf50=outvaltestmax;
ANNtestingtargetmax50=ANNtestingtarget50;
ANNmsetest50=mse(ANNOutf50,ANNtestingtargetmax50);
ANNmapetest50=mape(ANNOutf50,ANNtestingtargetmax50);
ANNmbetest50=mbe(ANNOutf50,ANNtestingtargetmax50);
ANNr2test50=rsquare(ANNOutf50,ANNtestingtargetmax50);
ANNperf50=[ANNmsetest50 ANNmapetest50 ANNmbetest50 ANNr2test50];
ANNOutf50train=outvalmax;
ANNPtestMax50=ANNPtest50'*maxx;
%
height50=[10 20 30 40 50];
rang50=[0 13];
rl50=[1:13];
testingtarget50=ANNtestingtarget50;
figure

plot([ANNPtestMax50'; ANNOutf50' ],height50);
xlim(rang50)
%ylim([-0.4 0.8])
title(['ANN 50m Testing MSE=' num2str(ANNmsetest50) ',MAPE=' num2str(ANNmapetest50) ',MBE=' num2str(ANNmbetest50)  ',R^2=' num2str(ANNr2test50)]);
%camroll(90)
% [10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180]
%
figure
%rl=[1:13];
plot( ANNOutf50, ANNtestingtarget50,'ob',rl50,rl50,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl50)+0.5]);
ylim([0 max(rl50)+0.5]);
title(['ANN 50 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test50*perc,2)) ' %'])

figure
interv=1:200;
plot( [testingtarget50(interv) ],'b');
hold on
plot( [ANNOutf50(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')
meantarget50=mean([seriesT(:,1:inputsize)'*maxx; testingtarget50' ]');
meanANN50=mean([ANNPtestMax50'; ANNOutf50' ]');
figure
plot(meantarget50,height50,'k');
hold on
plot(meanANN50,height50,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('50 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf50]

%% 60


nex=2;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y60 = trainingtarget';
Ytest60 = testingtarget';
testingtarget60=Ytest60'*maxx;


%
%Create NN
height60=[height50 60];
mxr=12.5+nex*0.5;
rang60=[0 mxr];

%
rl60=[1:mxr];

%
ANNP60= [ANNP50; ANNOutf50train'/maxx];
ANNY60 = trainingtarget';
ANNPtest60 = [ANNPtest50; ANNOutf50'/maxx];
ANNYtest60 = testingtarget';
ANNtestingtarget60=ANNYtest60'*maxx;

%Create NN

numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP60)); %training_index

netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
netANN.trainFcn = 'traingda';
netANN.trainParam.epochs=numiter;
[netANN,tr,Y,E] = train(netANN,ANNP60,ANNY60);

outval = netANN(ANNP60);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest60);
outvaltestmax=outvaltest'*maxx;
ANNOutf60=outvaltestmax;
ANNtestingtargetmax60=testingtarget60;
ANNmsetest60=mse(ANNOutf60,ANNtestingtargetmax60);
ANNmapetest60=mape(ANNOutf60,ANNtestingtargetmax60);
ANNmbetest60=mbe(ANNOutf60,ANNtestingtargetmax60);
ANNr2test60=rsquare(ANNOutf60,ANNtestingtargetmax60);
ANNperf60=[ANNmsetest60 ANNmapetest60 ANNmbetest60 ANNr2test60];
ANNOutf60train=outvalmax;
%
figure
ANNPtestMax60=ANNPtest60'*maxx;

plot([ANNPtestMax60'; ANNOutf60' ],height60);
xlim(rang60)
%ylim([-0.4 0.8])
title(['ANN 60m Testing MSE=' num2str(ANNmsetest60) ',MAPE=' num2str(ANNmapetest60) ',MBE=' num2str(ANNmbetest60)  ',R^2=' num2str(ANNr2test60)]);
%camroll(60)
% [10 20 30 40 60 60 60 60 60 60 60 60 60 60 160 160 160 160]
%
figure
%rl=[1:13];
plot( ANNOutf60, ANNtestingtargetmax60,'ob',rl60,rl60,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl60)+0.5]);
ylim([0 max(rl60)+0.5]);
title(['ANN 60 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test60*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget60(interv) ],'b');
hold on
plot( [ANNOutf60(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget60=[meantarget50 mean(testingtarget60)];
meanANN60=[meanANN50 mean(ANNOutf60)];

figure
plot(meantarget60,height60,'k');
hold on
plot(meanANN60,height60,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('60 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf60]

%% 70


nex=3;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y70 = trainingtarget';
Ytest70 = testingtarget';
testingtarget70=Ytest70'*maxx;


%
%Create NN
height70=[height60 70];
mxr=12.5+nex*0.5;
rang70=[0 mxr];

%
rl70=[1:mxr];

%
ANNP70= [ANNP60; ANNOutf60train'/maxx];
ANNY70 = trainingtarget';
ANNPtest70 = [ANNPtest60; ANNOutf60'/maxx];
ANNYtest70 = testingtarget';
ANNtestingtarget70=ANNYtest70'*maxx;

%Create NN


numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP70)); %training_index
netANN.trainParam.epochs=numiter;
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
netANN.trainFcn = 'traingda';
netANN.trainParam.epochs=numiter;
[netANN,tr,Y,E] = train(netANN,ANNP70,ANNY70);

outval = netANN(ANNP70);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest70);
outvaltestmax=outvaltest'*maxx;
ANNOutf70=outvaltestmax;
ANNtestingtargetmax70=testingtarget70;
ANNmsetest70=mse(ANNOutf70,ANNtestingtargetmax70);
ANNmapetest70=mape(ANNOutf70,ANNtestingtargetmax70);
ANNmbetest70=mbe(ANNOutf70,ANNtestingtargetmax70);
ANNr2test70=rsquare(ANNOutf70,ANNtestingtargetmax70);
ANNperf70=[ANNmsetest70 ANNmapetest70 ANNmbetest70 ANNr2test70];
ANNOutf70train=outvalmax;
%
figure
ANNPtestMax70=ANNPtest70'*maxx;

plot([ANNPtestMax70'; ANNOutf70' ],height70);
xlim(rang70)
%ylim([-0.4 0.8])
title(['ANN 70m Testing MSE=' num2str(ANNmsetest70) ',MAPE=' num2str(ANNmapetest70) ',MBE=' num2str(ANNmbetest70)  ',R^2=' num2str(ANNr2test70)]);
%camroll(70)
% [10 20 30 40 70 70 70 70 70 70 70 70 70 70 170 170 170 170]
%
figure
%rl=[1:13];
plot( ANNOutf70, ANNtestingtargetmax70,'ob',rl70,rl70,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl70)+0.5]);
ylim([0 max(rl70)+0.5]);
title(['ANN 70 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test70*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget70(interv) ],'b');
hold on
plot( [ANNOutf70(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget70=[meantarget60 mean(testingtarget70)];
meanANN70=[meanANN60 mean(ANNOutf70)];

figure
plot(meantarget70,height70,'k');
hold on
plot(meanANN70,height70,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('70 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf70]
%%
%% 80


nex=4;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y80 = trainingtarget';
Ytest80 = testingtarget';
testingtarget80=Ytest80'*maxx;


%
%Create NN
height80=[height70 80];
mxr=12.5+nex*0.5;
rang80=[0 mxr];

%
rl80=[1:mxr];

%
ANNP80= [ANNP70; ANNOutf70train'/maxx];
ANNY80 = trainingtarget';
ANNPtest80 = [ANNPtest70; ANNOutf70'/maxx];
ANNYtest80 = testingtarget';
ANNtestingtarget80=ANNYtest80'*maxx;

%Create NN

numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP80)); %training_index
netANN.trainParam.epochs=numiter;
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
netANN.trainFcn = 'traingda';
netANN.trainParam.epochs=numiter;

[netANN,tr,Y,E] = train(netANN,ANNP80,ANNY80);

outval = netANN(ANNP80);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest80);
outvaltestmax=outvaltest'*maxx;
ANNOutf80=outvaltestmax;
ANNtestingtargetmax80=testingtarget80;
ANNmsetest80=mse(ANNOutf80,ANNtestingtargetmax80);
ANNmapetest80=mape(ANNOutf80,ANNtestingtargetmax80);
ANNmbetest80=mbe(ANNOutf80,ANNtestingtargetmax80);
ANNr2test80=rsquare(ANNOutf80,ANNtestingtargetmax80);
ANNperf80=[ANNmsetest80 ANNmapetest80 ANNmbetest80 ANNr2test80];
ANNOutf80train=outvalmax;
%
figure
ANNPtestMax80=ANNPtest80'*maxx;

plot([ANNPtestMax80'; ANNOutf80' ],height80);
xlim(rang80)
%ylim([-0.4 0.8])
title(['ANN 80m Testing MSE=' num2str(ANNmsetest80) ',MAPE=' num2str(ANNmapetest80) ',MBE=' num2str(ANNmbetest80)  ',R^2=' num2str(ANNr2test80)]);
%camroll(80)
% [10 20 30 40 80 80 80 80 80 80 80 80 80 80 180 180 180 180]
%
figure
%rl=[1:13];
plot( ANNOutf80, ANNtestingtargetmax80,'ob',rl80,rl80,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl80)+0.5]);
ylim([0 max(rl80)+0.5]);
title(['ANN 80 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test80*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget80(interv) ],'b');
hold on
plot( [ANNOutf80(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget80=[meantarget70 mean(testingtarget80)];
meanANN80=[meanANN70 mean(ANNOutf80)];

figure
plot(meantarget80,height80,'k');
hold on
plot(meanANN80,height80,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('80 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf80]

%% 90


nex=5;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y90 = trainingtarget';
Ytest90 = testingtarget';
testingtarget90=Ytest90'*maxx;


%
%Create NN
height90=[height80 90];
mxr=12.5+nex*0.5;
rang90=[0 mxr];

%
rl90=[1:mxr];

%
ANNP90= [ANNP80; ANNOutf80train'/maxx];
ANNY90 = trainingtarget';
ANNPtest90 = [ANNPtest80; ANNOutf80'/maxx];
ANNYtest90 = testingtarget';
ANNtestingtarget90=ANNYtest90'*maxx;

%Create NN

numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP90)); %training_index
netANN.trainParam.epochs=numiter;
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
netANN.trainFcn = 'traingda';
netANN.trainParam.epochs=numiter;
[netANN,tr,Y,E] = train(netANN,ANNP90,ANNY90);

outval = netANN(ANNP90);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest90);
outvaltestmax=outvaltest'*maxx;
ANNOutf90=outvaltestmax;
ANNtestingtargetmax90=testingtarget90;
ANNmsetest90=mse(ANNOutf90,ANNtestingtargetmax90);
ANNmapetest90=mape(ANNOutf90,ANNtestingtargetmax90);
ANNmbetest90=mbe(ANNOutf90,ANNtestingtargetmax90);
ANNr2test90=rsquare(ANNOutf90,ANNtestingtargetmax90);
ANNperf90=[ANNmsetest90 ANNmapetest90 ANNmbetest90 ANNr2test90];
ANNOutf90train=outvalmax;
%
figure
ANNPtestMax90=ANNPtest90'*maxx;

plot([ANNPtestMax90'; ANNOutf90' ],height90);
xlim(rang90)
%ylim([-0.4 0.8])
title(['ANN 90m Testing MSE=' num2str(ANNmsetest90) ',MAPE=' num2str(ANNmapetest90) ',MBE=' num2str(ANNmbetest90)  ',R^2=' num2str(ANNr2test90)]);
%camroll(90)
% [10 20 30 40 90 90 90 90 90 90 90 90 90 90 190 190 190 190]
%
figure
%rl=[1:13];
plot( ANNOutf90, ANNtestingtargetmax90,'ob',rl90,rl90,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl90)+0.5]);
ylim([0 max(rl90)+0.5]);
title(['ANN 90 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test90*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget90(interv) ],'b');
hold on
plot( [ANNOutf90(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget90=[meantarget80 mean(testingtarget90)];
meanANN90=[meanANN80 mean(ANNOutf90)];

figure
plot(meantarget90,height90,'k');
hold on
plot(meanANN90,height90,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('90 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf90]

%% 100


nex=6;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y100 = trainingtarget';
Ytest100 = testingtarget';
testingtarget100=Ytest100'*maxx;


%
%Create NN
height100=[height90 100];
mxr=12.5+nex*0.5;
rang100=[0 mxr];

%
rl100=[1:mxr];

%
ANNP100= [ANNP90; ANNOutf90train'/maxx];
ANNY100 = trainingtarget';
ANNPtest100 = [ANNPtest90; ANNOutf90'/maxx];
ANNYtest100 = testingtarget';
ANNtestingtarget100=ANNYtest100'*maxx;

%Create NN

numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP100)); %training_index
netANN.trainParam.epochs=numiter;
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
netANN.trainFcn = 'traingda';
netANN.trainParam.epochs=numiter;
[netANN,tr,Y,E] = train(netANN,ANNP100,ANNY100);

outval = netANN(ANNP100);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest100);
outvaltestmax=outvaltest'*maxx;
ANNOutf100=outvaltestmax;
ANNtestingtargetmax100=testingtarget100;
ANNmsetest100=mse(ANNOutf100,ANNtestingtargetmax100);
ANNmapetest100=mape(ANNOutf100,ANNtestingtargetmax100);
ANNmbetest100=mbe(ANNOutf100,ANNtestingtargetmax100);
ANNr2test100=rsquare(ANNOutf100,ANNtestingtargetmax100);
ANNperf100=[ANNmsetest100 ANNmapetest100 ANNmbetest100 ANNr2test100];
ANNOutf100train=outvalmax;
%
figure
ANNPtestMax100=ANNPtest100'*maxx;

plot([ANNPtestMax100'; ANNOutf100' ],height100);
xlim(rang100)
%ylim([-0.4 0.8])
title(['ANN 100m Testing MSE=' num2str(ANNmsetest100) ',MAPE=' num2str(ANNmapetest100) ',MBE=' num2str(ANNmbetest100)  ',R^2=' num2str(ANNr2test100)]);
%camroll(100)
% [10 20 30 40 100 100 100 100 100 100 100 100 100 100 1100 1100 1100 1100]
%
figure
%rl=[1:13];
plot( ANNOutf100, ANNtestingtargetmax100,'ob',rl100,rl100,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl100)+0.5]);
ylim([0 max(rl100)+0.5]);
title(['ANN 100 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test100*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget100(interv) ],'b');
hold on
plot( [ANNOutf100(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget100=[meantarget90 mean(testingtarget100)];
meanANN100=[meanANN90 mean(ANNOutf100)];

figure
plot(meantarget100,height100,'k');
hold on
plot(meanANN100,height100,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('100 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf100]

%% 110


nex=7;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y110 = trainingtarget';
Ytest110 = testingtarget';
testingtarget110=Ytest110'*maxx;


%
%Create NN
height110=[height100 110];
mxr=12.5+nex*0.5;
rang110=[0 mxr];

%
rl110=[1:mxr];

%
ANNP110= [ANNP100; ANNOutf100train'/maxx];
ANNY110 = trainingtarget';
ANNPtest110 = [ANNPtest100; ANNOutf100'/maxx];
ANNYtest110 = testingtarget';
ANNtestingtarget110=ANNYtest110'*maxx;

%Create NN

numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP110)); %training_index
netANN.trainParam.epochs=numiter;
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
netANN.trainFcn = 'traingda';
netANN.trainParam.epochs=numiter;
[netANN,tr,Y,E] = train(netANN,ANNP110,ANNY110);

outval = netANN(ANNP110);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest110);
outvaltestmax=outvaltest'*maxx;
ANNOutf110=outvaltestmax;
ANNtestingtargetmax110=testingtarget110;
ANNmsetest110=mse(ANNOutf110,ANNtestingtargetmax110);
ANNmapetest110=mape(ANNOutf110,ANNtestingtargetmax110);
ANNmbetest110=mbe(ANNOutf110,ANNtestingtargetmax110);
ANNr2test110=rsquare(ANNOutf110,ANNtestingtargetmax110);
ANNperf110=[ANNmsetest110 ANNmapetest110 ANNmbetest110 ANNr2test110];
ANNOutf110train=outvalmax;
%
figure
ANNPtestMax110=ANNPtest110'*maxx;

plot([ANNPtestMax110'; ANNOutf110' ],height110);
xlim(rang110)
%ylim([-0.4 0.8])
title(['ANN 110m Testing MSE=' num2str(ANNmsetest110) ',MAPE=' num2str(ANNmapetest110) ',MBE=' num2str(ANNmbetest110)  ',R^2=' num2str(ANNr2test110)]);
%camroll(110)
% [10 20 30 40 110 110 110 110 110 110 110 110 110 110 1110 1110 1110 1110]
%
figure
%rl=[1:13];
plot( ANNOutf110, ANNtestingtargetmax110,'ob',rl110,rl110,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl110)+0.5]);
ylim([0 max(rl110)+0.5]);
title(['ANN 110 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test110*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget110(interv) ],'b');
hold on
plot( [ANNOutf110(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget110=[meantarget100 mean(testingtarget110)];
meanANN110=[meanANN100 mean(ANNOutf110)];

figure
plot(meantarget110,height110,'k');
hold on
plot(meanANN110,height110,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('110 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf110]

%% 120


nex=8;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y120 = trainingtarget';
Ytest120 = testingtarget';
testingtarget120=Ytest120'*maxx;


%
%Create NN
height120=[height110 120];
mxr=12.5+nex*0.5;
rang120=[0 mxr];

%
rl120=[1:mxr];

%
ANNP120= [ANNP110; ANNOutf110train'/maxx];
ANNY120 = trainingtarget';
ANNPtest120 = [ANNPtest110; ANNOutf110'/maxx];
ANNYtest120 = testingtarget';
ANNtestingtarget120=ANNYtest120'*maxx;

%Create NN

numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP120)); %training_index
netANN.trainParam.epochs=numiter;
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
netANN.trainFcn = 'traingda';
netANN.trainParam.epochs=numiter;
[netANN,tr,Y,E] = train(netANN,ANNP120,ANNY120);

outval = netANN(ANNP120);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest120);
outvaltestmax=outvaltest'*maxx;
ANNOutf120=outvaltestmax;
ANNtestingtargetmax120=testingtarget120;
ANNmsetest120=mse(ANNOutf120,ANNtestingtargetmax120);
ANNmapetest120=mape(ANNOutf120,ANNtestingtargetmax120);
ANNmbetest120=mbe(ANNOutf120,ANNtestingtargetmax120);
ANNr2test120=rsquare(ANNOutf120,ANNtestingtargetmax120);
ANNperf120=[ANNmsetest120 ANNmapetest120 ANNmbetest120 ANNr2test120];
ANNOutf120train=outvalmax;
%
figure
ANNPtestMax120=ANNPtest120'*maxx;

plot([ANNPtestMax120'; ANNOutf120' ],height120);
xlim(rang120)
%ylim([-0.4 0.8])
title(['ANN 120m Testing MSE=' num2str(ANNmsetest120) ',MAPE=' num2str(ANNmapetest120) ',MBE=' num2str(ANNmbetest120)  ',R^2=' num2str(ANNr2test120)]);
%camroll(120)
% [10 20 30 40 120 120 120 120 120 120 120 120 120 120 1120 1120 1120 1120]
%
figure
%rl=[1:13];
plot( ANNOutf120, ANNtestingtargetmax120,'ob',rl120,rl120,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl120)+0.5]);
ylim([0 max(rl120)+0.5]);
title(['ANN 120 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test120*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget120(interv) ],'b');
hold on
plot( [ANNOutf120(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget120=[meantarget110 mean(testingtarget120)];
meanANN120=[meanANN110 mean(ANNOutf120)];

figure
plot(meantarget120,height120,'k');
hold on
plot(meanANN120,height120,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('120 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf120]
ANN_perf_all=[ANNperf50; ANNperf60; ANNperf70; ANNperf80; ANNperf90; ANNperf100; ANNperf110; ANNperf120];
%% 130


nex=9;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y130 = trainingtarget';
Ytest130 = testingtarget';
testingtarget130=Ytest130'*maxx;


%
%Create NN
height130=[height120 130];
mxr=12.5+nex*0.5;
rang130=[0 mxr];

%
rl130=[1:mxr];

%
ANNP130= [ANNP120; ANNOutf120train'/maxx];
ANNY130 = trainingtarget';
ANNPtest130 = [ANNPtest120; ANNOutf120'/maxx];
ANNYtest130 = testingtarget';
ANNtestingtarget130=ANNYtest130'*maxx;

%Create NN

numiter=20;
numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP130)); %training_index
netANN.trainParam.epochs=numiter;
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
[netANN,tr,Y,E] = train(netANN,ANNP130,ANNY130);

outval = netANN(ANNP130);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest130);
outvaltestmax=outvaltest'*maxx;
ANNOutf130=outvaltestmax;
ANNtestingtargetmax130=testingtarget130;
ANNmsetest130=mse(ANNOutf130,ANNtestingtargetmax130);
ANNmapetest130=mape(ANNOutf130,ANNtestingtargetmax130);
ANNmbetest130=mbe(ANNOutf130,ANNtestingtargetmax130);
ANNr2test130=rsquare(ANNOutf130,ANNtestingtargetmax130);
ANNperf130=[ANNmsetest130 ANNmapetest130 ANNmbetest130 ANNr2test130];
ANNOutf130train=outvalmax;
%
figure
ANNPtestMax130=ANNPtest130'*maxx;

plot([ANNPtestMax130'; ANNOutf130' ],height130);
xlim(rang130)
%ylim([-0.4 0.8])
title(['ANN 130m Testing MSE=' num2str(ANNmsetest130) ',MAPE=' num2str(ANNmapetest130) ',MBE=' num2str(ANNmbetest130)  ',R^2=' num2str(ANNr2test130)]);
%camroll(130)
% [10 20 30 40 130 130 130 130 130 130 130 130 130 130 1130 1130 1130 1130]
%
figure
%rl=[1:13];
plot( ANNOutf130, ANNtestingtargetmax130,'ob',rl130,rl130,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl130)+0.5]);
ylim([0 max(rl130)+0.5]);
title(['ANN 130 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test130*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget130(interv) ],'b');
hold on
plot( [ANNOutf130(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget130=[meantarget120 mean(testingtarget130)];
meanANN130=[meanANN120 mean(ANNOutf130)];

figure
plot(meantarget130,height130,'k');
hold on
plot(meanANN130,height130,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('130 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf130]

%% 140


nex=10;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y140 = trainingtarget';
Ytest140 = testingtarget';
testingtarget140=Ytest140'*maxx;


%
%Create NN
height140=[height130 140];
mxr=12.5+nex*0.5;
rang140=[0 mxr];

%
rl140=[1:mxr];

%
ANNP140= [ANNP130; ANNOutf130train'/maxx];
ANNY140 = trainingtarget';
ANNPtest140 = [ANNPtest130; ANNOutf130'/maxx];
ANNYtest140 = testingtarget';
ANNtestingtarget140=ANNYtest140'*maxx;

%Create NN

numiter=20;
numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP140)); %training_index
netANN.trainParam.epochs=numiter;
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
[netANN,tr,Y,E] = train(netANN,ANNP140,ANNY140);

outval = netANN(ANNP140);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest140);
outvaltestmax=outvaltest'*maxx;
ANNOutf140=outvaltestmax;
ANNtestingtargetmax140=testingtarget140;
ANNmsetest140=mse(ANNOutf140,ANNtestingtargetmax140);
ANNmapetest140=mape(ANNOutf140,ANNtestingtargetmax140);
ANNmbetest140=mbe(ANNOutf140,ANNtestingtargetmax140);
ANNr2test140=rsquare(ANNOutf140,ANNtestingtargetmax140);
ANNperf140=[ANNmsetest140 ANNmapetest140 ANNmbetest140 ANNr2test140];
ANNOutf140train=outvalmax;
%
figure
ANNPtestMax140=ANNPtest140'*maxx;

plot([ANNPtestMax140'; ANNOutf140' ],height140);
xlim(rang140)
%ylim([-0.4 0.8])
title(['ANN 140m Testing MSE=' num2str(ANNmsetest140) ',MAPE=' num2str(ANNmapetest140) ',MBE=' num2str(ANNmbetest140)  ',R^2=' num2str(ANNr2test140)]);
%camroll(140)
% [10 20 30 40 140 140 140 140 140 140 140 140 140 140 1140 1140 1140 1140]
%
figure
%rl=[1:13];
plot( ANNOutf140, ANNtestingtargetmax140,'ob',rl140,rl140,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl140)+0.5]);
ylim([0 max(rl140)+0.5]);
title(['ANN 140 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test140*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget140(interv) ],'b');
hold on
plot( [ANNOutf140(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget140=[meantarget130 mean(testingtarget140)];
meanANN140=[meanANN130 mean(ANNOutf140)];

figure
plot(meantarget140,height140,'k');
hold on
plot(meanANN140,height140,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('140 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf140]

%% 150


nex=11;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y150 = trainingtarget';
Ytest150 = testingtarget';
testingtarget150=Ytest150'*maxx;


%
%Create NN
height150=[height140 150];
mxr=12.5+nex*0.5;
rang150=[0 mxr];

%
rl150=[1:mxr];

%
ANNP150= [ANNP140; ANNOutf140train'/maxx];
ANNY150 = trainingtarget';
ANNPtest150 = [ANNPtest140; ANNOutf140'/maxx];
ANNYtest150 = testingtarget';
ANNtestingtarget150=ANNYtest150'*maxx;

%Create NN

numiter=20;
numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP150)); %training_index
netANN.trainParam.epochs=numiter;
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
[netANN,tr,Y,E] = train(netANN,ANNP150,ANNY150);

outval = netANN(ANNP150);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest150);
outvaltestmax=outvaltest'*maxx;
ANNOutf150=outvaltestmax;
ANNtestingtargetmax150=testingtarget150;
ANNmsetest150=mse(ANNOutf150,ANNtestingtargetmax150);
ANNmapetest150=mape(ANNOutf150,ANNtestingtargetmax150);
ANNmbetest150=mbe(ANNOutf150,ANNtestingtargetmax150);
ANNr2test150=rsquare(ANNOutf150,ANNtestingtargetmax150);
ANNperf150=[ANNmsetest150 ANNmapetest150 ANNmbetest150 ANNr2test150];
ANNOutf150train=outvalmax;
%
figure
ANNPtestMax150=ANNPtest150'*maxx;

plot([ANNPtestMax150'; ANNOutf150' ],height150);
xlim(rang150)
%ylim([-0.4 0.8])
title(['ANN 150m Testing MSE=' num2str(ANNmsetest150) ',MAPE=' num2str(ANNmapetest150) ',MBE=' num2str(ANNmbetest150)  ',R^2=' num2str(ANNr2test150)]);
%camroll(150)
% [10 20 30 40 150 150 150 150 150 150 150 150 150 150 1150 1150 1150 1150]
%
figure
%rl=[1:13];
plot( ANNOutf150, ANNtestingtargetmax150,'ob',rl150,rl150,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl150)+0.5]);
ylim([0 max(rl150)+0.5]);
title(['ANN 150 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test150*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget150(interv) ],'b');
hold on
plot( [ANNOutf150(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget150=[meantarget140 mean(testingtarget150)];
meanANN150=[meanANN140 mean(ANNOutf150)];

figure
plot(meantarget150,height150,'k');
hold on
plot(meanANN150,height150,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('150 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf150]

%% 160


nex=12;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y160 = trainingtarget';
Ytest160 = testingtarget';
testingtarget160=Ytest160'*maxx;


%
%Create NN
height160=[height150 160];
mxr=12.5+nex*0.5;
rang160=[0 mxr];

%
rl160=[1:mxr];

%
ANNP160= [ANNP150; ANNOutf150train'/maxx];
ANNY160 = trainingtarget';
ANNPtest160 = [ANNPtest150; ANNOutf150'/maxx];
ANNYtest160 = testingtarget';
ANNtestingtarget160=ANNYtest160'*maxx;

%Create NN

numiter=20;
numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP160)); %training_index
netANN.trainParam.epochs=numiter;
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
[netANN,tr,Y,E] = train(netANN,ANNP160,ANNY160);

outval = netANN(ANNP160);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest160);
outvaltestmax=outvaltest'*maxx;
ANNOutf160=outvaltestmax;
ANNtestingtargetmax160=testingtarget160;
ANNmsetest160=mse(ANNOutf160,ANNtestingtargetmax160);
ANNmapetest160=mape(ANNOutf160,ANNtestingtargetmax160);
ANNmbetest160=mbe(ANNOutf160,ANNtestingtargetmax160);
ANNr2test160=rsquare(ANNOutf160,ANNtestingtargetmax160);
ANNperf160=[ANNmsetest160 ANNmapetest160 ANNmbetest160 ANNr2test160];
ANNOutf160train=outvalmax;
%
figure
ANNPtestMax160=ANNPtest160'*maxx;

plot([ANNPtestMax160'; ANNOutf160' ],height160);
xlim(rang160)
%ylim([-0.4 0.8])
title(['ANN 160m Testing MSE=' num2str(ANNmsetest160) ',MAPE=' num2str(ANNmapetest160) ',MBE=' num2str(ANNmbetest160)  ',R^2=' num2str(ANNr2test160)]);
%camroll(160)
% [10 20 30 40 160 160 160 160 160 160 160 160 160 160 1160 1160 1160 1160]
%
figure
%rl=[1:13];
plot( ANNOutf160, ANNtestingtargetmax160,'ob',rl160,rl160,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl160)+0.5]);
ylim([0 max(rl160)+0.5]);
title(['ANN 160 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test160*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget160(interv) ],'b');
hold on
plot( [ANNOutf160(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget160=[meantarget150 mean(testingtarget160)];
meanANN160=[meanANN150 mean(ANNOutf160)];

figure
plot(meantarget160,height160,'k');
hold on
plot(meanANN160,height160,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('160 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf160]

%% 170


nex=13;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y170 = trainingtarget';
Ytest170 = testingtarget';
testingtarget170=Ytest170'*maxx;


%
%Create NN
height170=[height160 170];
mxr=12.5+nex*0.5;
rang170=[0 mxr];

%
rl170=[1:mxr];

%
ANNP170= [ANNP160; ANNOutf160train'/maxx];
ANNY170 = trainingtarget';
ANNPtest170 = [ANNPtest160; ANNOutf160'/maxx];
ANNYtest170 = testingtarget';
ANNtestingtarget170=ANNYtest170'*maxx;

%Create NN

numiter=20;
numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP170)); %training_index
netANN.trainParam.epochs=numiter;
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
[netANN,tr,Y,E] = train(netANN,ANNP170,ANNY170);

outval = netANN(ANNP170);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest170);
outvaltestmax=outvaltest'*maxx;
ANNOutf170=outvaltestmax;
ANNtestingtargetmax170=testingtarget170;
ANNmsetest170=mse(ANNOutf170,ANNtestingtargetmax170);
ANNmapetest170=mape(ANNOutf170,ANNtestingtargetmax170);
ANNmbetest170=mbe(ANNOutf170,ANNtestingtargetmax170);
ANNr2test170=rsquare(ANNOutf170,ANNtestingtargetmax170);
ANNperf170=[ANNmsetest170 ANNmapetest170 ANNmbetest170 ANNr2test170];
ANNOutf170train=outvalmax;
%
figure
ANNPtestMax170=ANNPtest170'*maxx;

plot([ANNPtestMax170'; ANNOutf170' ],height170);
xlim(rang170)
%ylim([-0.4 0.8])
title(['ANN 170m Testing MSE=' num2str(ANNmsetest170) ',MAPE=' num2str(ANNmapetest170) ',MBE=' num2str(ANNmbetest170)  ',R^2=' num2str(ANNr2test170)]);
%camroll(170)
% [10 20 30 40 170 170 170 170 170 170 170 170 170 170 1170 1170 1170 1170]
%
figure
%rl=[1:13];
plot( ANNOutf170, ANNtestingtargetmax170,'ob',rl170,rl170,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl170)+0.5]);
ylim([0 max(rl170)+0.5]);
title(['ANN 170 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test170*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget170(interv) ],'b');
hold on
plot( [ANNOutf170(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget170=[meantarget160 mean(testingtarget170)];
meanANN170=[meanANN160 mean(ANNOutf170)];

figure
plot(meantarget170,height170,'k');
hold on
plot(meanANN170,height170,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('170 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf170]


%% 180


nex=14;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y180 = trainingtarget';
Ytest180 = testingtarget';
testingtarget180=Ytest180'*maxx;


%
%Create NN
height180=[height170 180];
mxr=12.5+nex*0.5;
rang180=[0 mxr];

%
rl180=[1:mxr];

%
ANNP180= [ANNP170; ANNOutf170train'/maxx];
ANNY180 = trainingtarget';
ANNPtest180 = [ANNPtest170; ANNOutf170'/maxx];
ANNYtest180 = testingtarget';
ANNtestingtarget180=ANNYtest180'*maxx;

%Create NN

numiter=20;
numhid=20;
netANN=feedforwardnet(numhid);
netANN.divideFcn='divideind';
%netANN.trainParam.showWindow = 0;
netANN.divideParam.trainInd=1:max(size(ANNP180)); %training_index
netANN.trainParam.epochs=numiter;
netANN.trainParam.mu=3;
netANN.trainParam.mu_dec=0.1;
netANN.trainParam.mu_inc =10;
[netANN,tr,Y,E] = train(netANN,ANNP180,ANNY180);

outval = netANN(ANNP180);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netANN(ANNPtest180);
outvaltestmax=outvaltest'*maxx;
ANNOutf180=outvaltestmax;
ANNtestingtargetmax180=testingtarget180;
ANNmsetest180=mse(ANNOutf180,ANNtestingtargetmax180);
ANNmapetest180=mape(ANNOutf180,ANNtestingtargetmax180);
ANNmbetest180=mbe(ANNOutf180,ANNtestingtargetmax180);
ANNr2test180=rsquare(ANNOutf180,ANNtestingtargetmax180);
ANNperf180=[ANNmsetest180 ANNmapetest180 ANNmbetest180 ANNr2test180];
ANNOutf180train=outvalmax;
%
figure
ANNPtestMax180=ANNPtest180'*maxx;

plot([ANNPtestMax180'; ANNOutf180' ],height180);
xlim(rang180)
%ylim([-0.4 0.8])
title(['ANN 180m Testing MSE=' num2str(ANNmsetest180) ',MAPE=' num2str(ANNmapetest180) ',MBE=' num2str(ANNmbetest180)  ',R^2=' num2str(ANNr2test180)]);
%camroll(180)
% [10 20 30 40 180 180 180 180 180 180 180 180 180 180 1180 1180 1180 1180]
%
figure
%rl=[1:13];
plot( ANNOutf180, ANNtestingtargetmax180,'ob',rl180,rl180,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl180)+0.5]);
ylim([0 max(rl180)+0.5]);
title(['ANN 180 m Testing ']);
text(2,12,['R^2=' num2str(round(ANNr2test180*perc,2)) ' %'])



figure
interv=1:200;
plot( [testingtarget180(interv) ],'b');
hold on
plot( [ANNOutf180(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','ANN est','Location','northwest')

meantarget180=[meantarget170 mean(testingtarget180)];
meanANN180=[meanANN170 mean(ANNOutf180)];

figure
plot(meantarget180,height180,'k');
hold on
plot(meanANN180,height180,'--r');

hold off
title('average')
legend('measured','ANN est','Location','northwest')

display ('180 m Testing:     MSE     MAPE        MBE        R2')
[ANNperf180]
ANN_perf_all=[ANNperf50; ANNperf60; ANNperf70; ANNperf80; ANNperf90; ANNperf100; ANNperf110; ANNperf120; ANNperf130; ANNperf140; ANNperf150; ANNperf160; ANNperf170; ANNperf180];
%ANN_perf_all=[ANNperf50; ANNperf60; ANNperf70; ANNperf80; ANNperf90; ANNperf100; ANNperf110; ANNperf120]