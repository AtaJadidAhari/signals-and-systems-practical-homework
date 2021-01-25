%% 
[y,Fs] = audioread('intro.wav');
sound(y,Fs);
z=y(:,1);
x = zeros(1,length(y));
for i=1:length(x)
    x(i)=y(i);
end
%%

n0=3000;
a=0.9995;
for i=n0+1:length(x)
    for j=1:n0
        x(i)= y(i)+(a^j)*y(i-j);
    end
end
%%
subplot(2,1,1)
plot(x)
subplot(2,1,2)
plot(y)
sound(x,Fs);
%% ztrans
sys=tf(x(10000:12000),1,[],'Variable','z^-1');
figure()
bode(sys)
%%
sys2=tf(y(10000:12000)',1,[],'Variable','z^-1');
figure()
bode(sys2)
%% 
sound(y,Fs);
%%

b=1;
a=[1 -a];
y2=filter(a,b,x);
sound(y2,Fs)
subplot(2,1,1)
plot(y)
subplot(2,1,2)
plot(y2)




