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

P50 = traininginput';
Y50 = trainingtarget';
Ptest50 = testinginput';
Ytest50 = testingtarget';
testingtarget50=Ytest50'*maxx;
%
%Create NN

%outval = netMLP(P);

trainingtargetmax=trainingtarget*maxx;

height50=[10 20 30 40 50];
rang50=[0 13];
rl50=[1:13];
% 1/7 WSE

LogLawP50 = traininginput';
LogLawY50 = trainingtarget';
LogLawPtest50 = testinginput';
LogLawYtest50 = testingtarget';
LogLawtestingtarget50=LogLawYtest50'*maxx;


alpha=1/3;
outval=LogLawP50(4,:)*(log(50/alpha)/log(40/alpha));
outvalmax=outval*maxx;
LogLawOutf50train=outvalmax;

outvaltest=LogLawPtest50(4,:)*(log(50/alpha)/log(40/alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget50;
LogLawOutf50=outvaltestmax;
LogLawmsetest50=mse(LogLawOutf50,testingtarget50);
LogLawmapetest50=mape(LogLawOutf50,testingtarget50);
LogLawmbetest50=mbe(LogLawOutf50,testingtarget50);
LogLawr2test50=rsquare(LogLawOutf50,testingtarget50);
LogLawperf50=[LogLawmsetest50 LogLawmapetest50 LogLawmbetest50 LogLawr2test50];
LogLawPtestMax50=LogLawPtest50'*maxx;

meantarget50=mean([seriesT(:,1:inputsize)'*maxx; testingtarget50' ]');
meanLogLaw50=mean([LogLawPtestMax50'; LogLawOutf50' ]');
figure
plot(meantarget50,height50,'k');
hold on
plot(meanLogLaw50,height50,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('50 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf50]

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

testingtargetmax=testingtarget*maxx;
target60=testingtarget60;

%
height60=[height50 60];
mxr=12.5+nex*0.5;
rang60=[0 mxr];
rl60=[1:mxr];
% 1/7 WSE
%
LogLawP60 = [LogLawP50; LogLawOutf50train/maxx];
LogLawY60 = trainingtarget';
LogLawPtest60 = [LogLawPtest50; LogLawOutf50'/maxx];
LogLawYtest60 = testingtarget';
LogLawtestingtarget60=LogLawYtest60'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP60(indl,:)*(log(60/alpha)/log(50/alpha));
outvalmax=outval*maxx;
LogLawOutf60train=outvalmax;


outvaltest=LogLawPtest60(indl,:)*(log(60/alpha)/log(50/alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget60;
LogLawOutf60=outvaltestmax;
LogLawmsetest60=mse(LogLawOutf60,testingtarget60);
LogLawmapetest60=mape(LogLawOutf60,testingtarget60);
LogLawmbetest60=mbe(LogLawOutf60,testingtarget60);
LogLawr2test60=rsquare(LogLawOutf60,testingtarget60);
LogLawperf60=[LogLawmsetest60 LogLawmapetest60 LogLawmbetest60 LogLawr2test60];
LogLawPtestMax60=LogLawPtest60'*maxx;

meantarget60=[meantarget50 mean(testingtarget60)];
meanLogLaw60=[meanLogLaw50 mean(LogLawOutf60)];

figure
plot(meantarget60,height60,'k');
hold on
plot(meanLogLaw60,height60,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('60 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf60]

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

testingtargetmax=testingtarget*maxx;
target70=testingtarget70;

%
height70=[height60 70];
mxr=12.5+nex*0.5;
rang70=[0 mxr];
rl70=[1:mxr];
% 1/7 WSE
%
LogLawP70 = [LogLawP60; LogLawOutf60train/maxx];
LogLawY70 = trainingtarget';
LogLawPtest70 = [LogLawPtest60; LogLawOutf60'/maxx];
LogLawYtest70 = testingtarget';
LogLawtestingtarget70=LogLawYtest70'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP70(indl,:)*(log(70/alpha)/log(60/alpha));
outvalmax=outval*maxx;
LogLawOutf70train=outvalmax;


outvaltest=LogLawPtest70(indl,:)*(log(70/alpha)/log(60/alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget70;
LogLawOutf70=outvaltestmax;
LogLawmsetest70=mse(LogLawOutf70,testingtarget70);
LogLawmapetest70=mape(LogLawOutf70,testingtarget70);
LogLawmbetest70=mbe(LogLawOutf70,testingtarget70);
LogLawr2test70=rsquare(LogLawOutf70,testingtarget70);
LogLawperf70=[LogLawmsetest70 LogLawmapetest70 LogLawmbetest70 LogLawr2test70];
LogLawPtestMax70=LogLawPtest70'*maxx;

meantarget70=[meantarget60 mean(testingtarget70)];
meanLogLaw70=[meanLogLaw60 mean(LogLawOutf70)];

figure
plot(meantarget70,height70,'k');
hold on
plot(meanLogLaw70,height70,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('70 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf70]

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

testingtargetmax=testingtarget*maxx;
target80=testingtarget80;

%
height80=[height70 80];
mxr=12.5+nex*0.5;
rang80=[0 mxr];
rl80=[1:mxr];
% 1/7 WSE
%
LogLawP80 = [LogLawP70; LogLawOutf70train/maxx];
LogLawY80 = trainingtarget';
LogLawPtest80 = [LogLawPtest70; LogLawOutf70'/maxx];
LogLawYtest80 = testingtarget';
LogLawtestingtarget80=LogLawYtest80'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP80(indl,:)*(log(80/alpha)/log(70/alpha));
outvalmax=outval*maxx;
LogLawOutf80train=outvalmax;


outvaltest=LogLawPtest80(indl,:)*(log(80/alpha)/log(70/alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget80;
LogLawOutf80=outvaltestmax;
LogLawmsetest80=mse(LogLawOutf80,testingtarget80);
LogLawmapetest80=mape(LogLawOutf80,testingtarget80);
LogLawmbetest80=mbe(LogLawOutf80,testingtarget80);
LogLawr2test80=rsquare(LogLawOutf80,testingtarget80);
LogLawperf80=[LogLawmsetest80 LogLawmapetest80 LogLawmbetest80 LogLawr2test80];
LogLawPtestMax80=LogLawPtest80'*maxx;

meantarget80=[meantarget70 mean(testingtarget80)];
meanLogLaw80=[meanLogLaw70 mean(LogLawOutf80)];

figure
plot(meantarget80,height80,'k');
hold on
plot(meanLogLaw80,height80,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('80 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf80]

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

testingtargetmax=testingtarget*maxx;
target90=testingtarget90;

%
height90=[height80 90];
mxr=12.5+nex*0.5;
rang90=[0 mxr];
rl90=[1:mxr];
% 1/7 WSE
%
LogLawP90 = [LogLawP80; LogLawOutf80train/maxx];
LogLawY90 = trainingtarget';
LogLawPtest90 = [LogLawPtest80; LogLawOutf80'/maxx];
LogLawYtest90 = testingtarget';
LogLawtestingtarget90=LogLawYtest90'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP90(indl,:)*(log(90/alpha)/log(80/alpha));
outvalmax=outval*maxx;
LogLawOutf90train=outvalmax;


outvaltest=LogLawPtest90(indl,:)*(log(90/alpha)/log(80/alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget90;
LogLawOutf90=outvaltestmax;
LogLawmsetest90=mse(LogLawOutf90,testingtarget90);
LogLawmapetest90=mape(LogLawOutf90,testingtarget90);
LogLawmbetest90=mbe(LogLawOutf90,testingtarget90);
LogLawr2test90=rsquare(LogLawOutf90,testingtarget90);
LogLawperf90=[LogLawmsetest90 LogLawmapetest90 LogLawmbetest90 LogLawr2test90];
LogLawPtestMax90=LogLawPtest90'*maxx;

meantarget90=[meantarget80 mean(testingtarget90)];
meanLogLaw90=[meanLogLaw80 mean(LogLawOutf90)];

figure
plot(meantarget90,height90,'k');
hold on
plot(meanLogLaw90,height90,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('90 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf90]

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

testingtargetmax=testingtarget*maxx;
target100=testingtarget100;

%
height100=[height90 100];
mxr=12.5+nex*0.5;
rang100=[0 mxr];
rl100=[1:mxr];
% 1/7 WSE
%
LogLawP100 = [LogLawP90; LogLawOutf90train/maxx];
LogLawY100 = trainingtarget';
LogLawPtest100 = [LogLawPtest90; LogLawOutf90'/maxx];
LogLawYtest100 = testingtarget';
LogLawtestingtarget100=LogLawYtest100'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP100(indl,:)*(log(100/alpha)/log(90/alpha));
outvalmax=outval*maxx;
LogLawOutf100train=outvalmax;


outvaltest=LogLawPtest100(indl,:)*(log(100/alpha)/log(90/alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget100;
LogLawOutf100=outvaltestmax;
LogLawmsetest100=mse(LogLawOutf100,testingtarget100);
LogLawmapetest100=mape(LogLawOutf100,testingtarget100);
LogLawmbetest100=mbe(LogLawOutf100,testingtarget100);
LogLawr2test100=rsquare(LogLawOutf100,testingtarget100);
LogLawperf100=[LogLawmsetest100 LogLawmapetest100 LogLawmbetest100 LogLawr2test100];
LogLawPtestMax100=LogLawPtest100'*maxx;

meantarget100=[meantarget90 mean(testingtarget100)];
meanLogLaw100=[meanLogLaw90 mean(LogLawOutf100)];

figure
plot(meantarget100,height100,'k');
hold on
plot(meanLogLaw100,height100,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('100 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf100]

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

testingtargetmax=testingtarget*maxx;
target110=testingtarget110;

%
height110=[height100 110];
mxr=12.5+nex*0.5;
rang110=[0 mxr];
rl110=[1:mxr];
% 1/7 WSE
%
LogLawP110 = [LogLawP100; LogLawOutf100train/maxx];
LogLawY110 = trainingtarget';
LogLawPtest110 = [LogLawPtest100; LogLawOutf100'/maxx];
LogLawYtest110 = testingtarget';
LogLawtestingtarget110=LogLawYtest110'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP110(indl,:)*(log(110/alpha)/log(100/alpha));
outvalmax=outval*maxx;
LogLawOutf110train=outvalmax;


outvaltest=LogLawPtest110(indl,:)*(log(110/alpha)/log(100/alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget110;
LogLawOutf110=outvaltestmax;
LogLawmsetest110=mse(LogLawOutf110,testingtarget110);
LogLawmapetest110=mape(LogLawOutf110,testingtarget110);
LogLawmbetest110=mbe(LogLawOutf110,testingtarget110);
LogLawr2test110=rsquare(LogLawOutf110,testingtarget110);
LogLawperf110=[LogLawmsetest110 LogLawmapetest110 LogLawmbetest110 LogLawr2test110];
LogLawPtestMax110=LogLawPtest110'*maxx;

meantarget110=[meantarget100 mean(testingtarget110)];
meanLogLaw110=[meanLogLaw100 mean(LogLawOutf110)];

figure
plot(meantarget110,height110,'k');
hold on
plot(meanLogLaw110,height110,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('110 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf110]

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

testingtargetmax=testingtarget*maxx;
target120=testingtarget120;

%
height120=[height110 120];
mxr=12.5+nex*0.5;
rang120=[0 mxr];
rl120=[1:mxr];
% 1/7 WSE
%
LogLawP120 = [LogLawP110; LogLawOutf110train/maxx];
LogLawY120 = trainingtarget';
LogLawPtest120 = [LogLawPtest110; LogLawOutf110'/maxx];
LogLawYtest120 = testingtarget';
LogLawtestingtarget120=LogLawYtest120'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP120(indl,:)*(log(120/alpha)/log(110/alpha));
outvalmax=outval*maxx;
LogLawOutf120train=outvalmax;


outvaltest=LogLawPtest120(indl,:)*(log(120/alpha)/log(110/alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget120;
LogLawOutf120=outvaltestmax;
LogLawmsetest120=mse(LogLawOutf120,testingtarget120);
LogLawmapetest120=mape(LogLawOutf120,testingtarget120);
LogLawmbetest120=mbe(LogLawOutf120,testingtarget120);
LogLawr2test120=rsquare(LogLawOutf120,testingtarget120);
LogLawperf120=[LogLawmsetest120 LogLawmapetest120 LogLawmbetest120 LogLawr2test120];
LogLawPtestMax120=LogLawPtest120'*maxx;

meantarget120=[meantarget110 mean(testingtarget120)];
meanLogLaw120=[meanLogLaw110 mean(LogLawOutf120)];

figure
plot(meantarget120,height120,'k');
hold on
plot(meanLogLaw120,height120,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('120 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf120]
LogLaw_perf_all=[LogLawperf50; LogLawperf60; LogLawperf70; LogLawperf80; LogLawperf90; LogLawperf100; LogLawperf110; LogLawperf120];
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

testingtargetmax=testingtarget*maxx;
target130=testingtarget130;

%
height130=[height120 130];
mxr=12.5+nex*0.5;
rang130=[0 mxr];
rl130=[1:mxr];
% 1/7 WSE
%
LogLawP130 = [LogLawP120; LogLawOutf120train/maxx];
LogLawY130 = trainingtarget';
LogLawPtest130 = [LogLawPtest120; LogLawOutf120'/maxx];
LogLawYtest130 = testingtarget';
LogLawtestingtarget130=LogLawYtest130'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP130(indl,:)*((130/120)^(alpha));
outvalmax=outval*maxx;
LogLawOutf130train=outvalmax;


outvaltest=LogLawPtest130(indl,:)*((130/120)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget130;
LogLawOutf130=outvaltestmax;
LogLawmsetest130=mse(LogLawOutf130,testingtarget130);
LogLawmapetest130=mape(LogLawOutf130,testingtarget130);
LogLawmbetest130=mbe(LogLawOutf130,testingtarget130);
LogLawr2test130=rsquare(LogLawOutf130,testingtarget130);
LogLawperf130=[LogLawmsetest130 LogLawmapetest130 LogLawmbetest130 LogLawr2test130];
LogLawPtestMax130=LogLawPtest130'*maxx;

meantarget130=[meantarget120 mean(testingtarget130)];
meanLogLaw130=[meanLogLaw120 mean(LogLawOutf130)];

figure
plot(meantarget130,height130,'k');
hold on
plot(meanLogLaw130,height130,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('130 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf130]

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

testingtargetmax=testingtarget*maxx;
target140=testingtarget140;

%
height140=[height130 140];
mxr=12.5+nex*0.5;
rang140=[0 mxr];
rl140=[1:mxr];
% 1/7 WSE
%
LogLawP140 = [LogLawP130; LogLawOutf130train/maxx];
LogLawY140 = trainingtarget';
LogLawPtest140 = [LogLawPtest130; LogLawOutf130'/maxx];
LogLawYtest140 = testingtarget';
LogLawtestingtarget140=LogLawYtest140'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP140(indl,:)*((140/130)^(alpha));
outvalmax=outval*maxx;
LogLawOutf140train=outvalmax;


outvaltest=LogLawPtest140(indl,:)*((140/130)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget140;
LogLawOutf140=outvaltestmax;
LogLawmsetest140=mse(LogLawOutf140,testingtarget140);
LogLawmapetest140=mape(LogLawOutf140,testingtarget140);
LogLawmbetest140=mbe(LogLawOutf140,testingtarget140);
LogLawr2test140=rsquare(LogLawOutf140,testingtarget140);
LogLawperf140=[LogLawmsetest140 LogLawmapetest140 LogLawmbetest140 LogLawr2test140];
LogLawPtestMax140=LogLawPtest140'*maxx;

meantarget140=[meantarget130 mean(testingtarget140)];
meanLogLaw140=[meanLogLaw130 mean(LogLawOutf140)];

figure
plot(meantarget140,height140,'k');
hold on
plot(meanLogLaw140,height140,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('140 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf140]

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

testingtargetmax=testingtarget*maxx;
target150=testingtarget150;

%
height150=[height140 150];
mxr=12.5+nex*0.5;
rang150=[0 mxr];
rl150=[1:mxr];
% 1/7 WSE
%
LogLawP150 = [LogLawP140; LogLawOutf140train/maxx];
LogLawY150 = trainingtarget';
LogLawPtest150 = [LogLawPtest140; LogLawOutf140'/maxx];
LogLawYtest150 = testingtarget';
LogLawtestingtarget150=LogLawYtest150'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP150(indl,:)*((150/140)^(alpha));
outvalmax=outval*maxx;
LogLawOutf150train=outvalmax;


outvaltest=LogLawPtest150(indl,:)*((150/140)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget150;
LogLawOutf150=outvaltestmax;
LogLawmsetest150=mse(LogLawOutf150,testingtarget150);
LogLawmapetest150=mape(LogLawOutf150,testingtarget150);
LogLawmbetest150=mbe(LogLawOutf150,testingtarget150);
LogLawr2test150=rsquare(LogLawOutf150,testingtarget150);
LogLawperf150=[LogLawmsetest150 LogLawmapetest150 LogLawmbetest150 LogLawr2test150];
LogLawPtestMax150=LogLawPtest150'*maxx;

meantarget150=[meantarget140 mean(testingtarget150)];
meanLogLaw150=[meanLogLaw140 mean(LogLawOutf150)];

figure
plot(meantarget150,height150,'k');
hold on
plot(meanLogLaw150,height150,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('150 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf150]

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

testingtargetmax=testingtarget*maxx;
target160=testingtarget160;

%
height160=[height150 160];
mxr=12.5+nex*0.5;
rang160=[0 mxr];
rl160=[1:mxr];
% 1/7 WSE
%
LogLawP160 = [LogLawP150; LogLawOutf150train/maxx];
LogLawY160 = trainingtarget';
LogLawPtest160 = [LogLawPtest150; LogLawOutf150'/maxx];
LogLawYtest160 = testingtarget';
LogLawtestingtarget160=LogLawYtest160'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP160(indl,:)*((160/150)^(alpha));
outvalmax=outval*maxx;
LogLawOutf160train=outvalmax;


outvaltest=LogLawPtest160(indl,:)*((160/150)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget160;
LogLawOutf160=outvaltestmax;
LogLawmsetest160=mse(LogLawOutf160,testingtarget160);
LogLawmapetest160=mape(LogLawOutf160,testingtarget160);
LogLawmbetest160=mbe(LogLawOutf160,testingtarget160);
LogLawr2test160=rsquare(LogLawOutf160,testingtarget160);
LogLawperf160=[LogLawmsetest160 LogLawmapetest160 LogLawmbetest160 LogLawr2test160];
LogLawPtestMax160=LogLawPtest160'*maxx;

meantarget160=[meantarget150 mean(testingtarget160)];
meanLogLaw160=[meanLogLaw150 mean(LogLawOutf160)];

figure
plot(meantarget160,height160,'k');
hold on
plot(meanLogLaw160,height160,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('160 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf160]

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

testingtargetmax=testingtarget*maxx;
target170=testingtarget170;

%
height170=[height160 170];
mxr=12.5+nex*0.5;
rang170=[0 mxr];
rl170=[1:mxr];
% 1/7 WSE
%
LogLawP170 = [LogLawP160; LogLawOutf160train/maxx];
LogLawY170 = trainingtarget';
LogLawPtest170 = [LogLawPtest160; LogLawOutf160'/maxx];
LogLawYtest170 = testingtarget';
LogLawtestingtarget170=LogLawYtest170'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP170(indl,:)*((170/160)^(alpha));
outvalmax=outval*maxx;
LogLawOutf170train=outvalmax;


outvaltest=LogLawPtest170(indl,:)*((170/160)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget170;
LogLawOutf170=outvaltestmax;
LogLawmsetest170=mse(LogLawOutf170,testingtarget170);
LogLawmapetest170=mape(LogLawOutf170,testingtarget170);
LogLawmbetest170=mbe(LogLawOutf170,testingtarget170);
LogLawr2test170=rsquare(LogLawOutf170,testingtarget170);
LogLawperf170=[LogLawmsetest170 LogLawmapetest170 LogLawmbetest170 LogLawr2test170];
LogLawPtestMax170=LogLawPtest170'*maxx;

meantarget170=[meantarget160 mean(testingtarget170)];
meanLogLaw170=[meanLogLaw160 mean(LogLawOutf170)];

figure
plot(meantarget170,height170,'k');
hold on
plot(meanLogLaw170,height170,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('170 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf170]

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

testingtargetmax=testingtarget*maxx;
target180=testingtarget180;

%
height180=[height170 180];
mxr=12.5+nex*0.5;
rang180=[0 mxr];
rl180=[1:mxr];
% 1/7 WSE
%
LogLawP180 = [LogLawP170; LogLawOutf170train/maxx];
LogLawY180 = trainingtarget';
LogLawPtest180 = [LogLawPtest170; LogLawOutf170'/maxx];
LogLawYtest180 = testingtarget';
LogLawtestingtarget180=LogLawYtest180'*maxx;


alpha=1/3;
indl=nex+3;
outval=LogLawP180(indl,:)*((180/170)^(alpha));
outvalmax=outval*maxx;
LogLawOutf180train=outvalmax;


outvaltest=LogLawPtest180(indl,:)*((180/170)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget180;
LogLawOutf180=outvaltestmax;
LogLawmsetest180=mse(LogLawOutf180,testingtarget180);
LogLawmapetest180=mape(LogLawOutf180,testingtarget180);
LogLawmbetest180=mbe(LogLawOutf180,testingtarget180);
LogLawr2test180=rsquare(LogLawOutf180,testingtarget180);
LogLawperf180=[LogLawmsetest180 LogLawmapetest180 LogLawmbetest180 LogLawr2test180];
LogLawPtestMax180=LogLawPtest180'*maxx;

meantarget180=[meantarget170 mean(testingtarget180)];
meanLogLaw180=[meanLogLaw170 mean(LogLawOutf180)];

figure
plot(meantarget180,height180,'k');
hold on
plot(meanLogLaw180,height180,'-.g');

hold off
title('average')
legend('measured','1/7 est','Location','northwest')

display ('180 m Testing:     MSE     MAPE        MBE        R2')
[LogLawperf180]
LogLaw_perf_all=[LogLawperf50; LogLawperf60; LogLawperf70; LogLawperf80; LogLawperf90; LogLawperf100; LogLawperf110; LogLawperf120; LogLawperf130; LogLawperf140; LogLawperf150; LogLawperf160; LogLawperf170; LogLawperf180];