clear all
n=2:5;
%n=4;
 for a = 1:10
    M=2^n;
    mkdir('D:\学习资料\论文\神经网络——信号调制方式识别\仿真_02\data\64X64data\test',num2str(M));
    msg = randi(M,10000,1)-1 ;%信源
    msg_qam=pskmod(msg,M);
    %mkdir('D:\学习资料\论文\神经网络——信号调制方式识别\仿真\仿真练习\仿真联系\'+num2str(M)+'QAM');
 for snr = 1:5
         ynoisy = awgn(msg_qam,snr*5);   %加噪声 
         picture = scatterplot(ynoisy);
    
    saveas(gca,"D:\学习资料\论文\神经网络——信号调制方式识别\仿真_02\data\64X64data\test\"+num2str(M)+"\"+num2str(a)+"_1("+num2str(snr)+')'+'.jpg')
 end
 end
n=2;
 for a = 1:10
 M=2^n;
 mkdir('D:\学习资料\论文\神经网络——信号调制方式识别\仿真_02\data\64X64data\test',num2str(M));
 msg = randi(M,20000,1)-1 ;%信源
 msg_qam=qammod(msg,M).';
 for snr = 1:5
    ynoisy = awgn(msg_qam,snr*5);   %加噪声 
    picture = scatterplot(ynoisy);
    %dos('md 32QAM'); % 在当前路径下生成文件夹a
    saveas(gca,"D:\学习资料\论文\神经网络——信号调制方式识别\仿真_02\data\64X64data\test\"+num2str(M)+"\"+num2str(a)+"_2("+num2str(snr)+')'+'.jpg')
 end
 end
  %close all
  n=6;
 for a = 1:10
 M=2^n;
  mkdir('D:\学习资料\论文\神经网络——信号调制方式识别\仿真_02\data\64X64data\test',num2str(M));
 msg = randi(M,5000,1)-1 ;%信源
 msg_qam=qammod(msg,M).';
 for snr = 1:5
    ynoisy = awgn(msg_qam,snr*5);   %加噪声 
    picture = scatterplot(ynoisy);
    % dos('md 32QAM'); % 在当前路径下生成文件夹a
     saveas(gca,"D:\学习资料\论文\神经网络——信号调制方式识别\仿真_02\data\64X64data\test\"+num2str(M)+"\"+num2str(a)+"_3("+num2str(snr)+')'+'.jpg')
 end
 end
 close all
%
