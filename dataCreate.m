clear all
n=2:5;
%n=4;
 for a = 1:10
    M=2^n;
    mkdir('D:\ѧϰ����\����\�����硪���źŵ��Ʒ�ʽʶ��\����_02\data\64X64data\test',num2str(M));
    msg = randi(M,10000,1)-1 ;%��Դ
    msg_qam=pskmod(msg,M);
    %mkdir('D:\ѧϰ����\����\�����硪���źŵ��Ʒ�ʽʶ��\����\������ϰ\������ϵ\'+num2str(M)+'QAM');
 for snr = 1:5
         ynoisy = awgn(msg_qam,snr*5);   %������ 
         picture = scatterplot(ynoisy);
    
    saveas(gca,"D:\ѧϰ����\����\�����硪���źŵ��Ʒ�ʽʶ��\����_02\data\64X64data\test\"+num2str(M)+"\"+num2str(a)+"_1("+num2str(snr)+')'+'.jpg')
 end
 end
n=2;
 for a = 1:10
 M=2^n;
 mkdir('D:\ѧϰ����\����\�����硪���źŵ��Ʒ�ʽʶ��\����_02\data\64X64data\test',num2str(M));
 msg = randi(M,20000,1)-1 ;%��Դ
 msg_qam=qammod(msg,M).';
 for snr = 1:5
    ynoisy = awgn(msg_qam,snr*5);   %������ 
    picture = scatterplot(ynoisy);
    %dos('md 32QAM'); % �ڵ�ǰ·���������ļ���a
    saveas(gca,"D:\ѧϰ����\����\�����硪���źŵ��Ʒ�ʽʶ��\����_02\data\64X64data\test\"+num2str(M)+"\"+num2str(a)+"_2("+num2str(snr)+')'+'.jpg')
 end
 end
  %close all
  n=6;
 for a = 1:10
 M=2^n;
  mkdir('D:\ѧϰ����\����\�����硪���źŵ��Ʒ�ʽʶ��\����_02\data\64X64data\test',num2str(M));
 msg = randi(M,5000,1)-1 ;%��Դ
 msg_qam=qammod(msg,M).';
 for snr = 1:5
    ynoisy = awgn(msg_qam,snr*5);   %������ 
    picture = scatterplot(ynoisy);
    % dos('md 32QAM'); % �ڵ�ǰ·���������ļ���a
     saveas(gca,"D:\ѧϰ����\����\�����硪���źŵ��Ʒ�ʽʶ��\����_02\data\64X64data\test\"+num2str(M)+"\"+num2str(a)+"_3("+num2str(snr)+')'+'.jpg')
 end
 end
 close all
%
