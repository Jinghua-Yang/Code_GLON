clc; clear all; close all;
addpath(genpath(cd));
rand('seed',213412); 

EN_HaLRTC    = 1;
EN_tSVD      = 1;
EN_TMacTT    = 1;
EN_KBR       = 1;
EN_TT_TV     = 1;
EN_GLON      = 1; % Ours

methodname  = {'HaLRTC','tSVD','TMac-TT','KBR','TT-TV','GLON'};

X0   = double(imread('baboon.bmp'));
nway = size(X0);
X0   = min( 255, max( X0, 0 ));
name = {'baboon'};

for SR = [0.1]
%% Generate known data
Nway = size( image2vdt256(X0) );
P = round(SR*prod(Nway));      % prod返回乘积
Known = randsample(prod(Nway),P);
[Known,~] = sort(Known);

%% Ket Augmentation
Xtrue = image2vdt256(X0);
%% Missing data
Xkn          = Xtrue(Known);
Xmiss        = zeros(Nway);
Xmiss(Known) = Xkn;
Xmiss        = vdt2image256(Xmiss);
    
imname=[num2str(name{1}),'_tensor0','_SR_',num2str(SR),'.mat'];
save(imname,'X0','Xmiss','Xkn','Known');

%% HaLRTC
j = 1;
if EN_HaLRTC
    %%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);
    
    opts = [];
    alpha      = ones(ndims(X0),1);
    opts.alpha = alpha/sum(alpha); 
    opts.tol   = 1e-4; 
    opts.maxit = 1000; 
    opts.rho   = 1.1; 
    opts.max_beta = 1e10;
    opts.X0       = X0;
    
    beta = [5*1e-5, 1e-4, 5*1e-4, 1e-3, 5*1e-3];
    for n = 1:length(beta)
        opts.beta = beta(n);
        t0=tic;
        X = HaLRTC( Xkn, Known, opts );
        X    = min( 255, max( X, 0 ));
        time=toc(t0);
        for i=1:1:3
            PSNRvector(i)=psnr3(X0(:,:,i)/255,X(:,:,i)/255);
        end
        psnr = mean(PSNRvector);
        
        for i=1:1:3
            SSIMvector(i)=ssim3(X0(:,:,i),X(:,:,i));
        end
        ssim = mean(SSIMvector);
        
        display(sprintf('psnr=%.2f,ssim=%.4f,beta=%.5f', psnr, ssim, opts.beta))
        display(sprintf('=================================='))
        
        imname=[num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_beta_',num2str(opts.beta),'.mat'];
        save(imname,'X','time');
    end
end
%% tSVD
j = j+1;
if EN_tSVD
     %%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);
       
    Mask  = zeros( Nway );
    Mask( Known ) = 1;
    Mask = vdt2image256(Mask);
    
    alpha = 1; maxItr  = 1000; myNorm = 'tSVD_1'; 
    A     = diag(sparse(double(Mask(:)))); 
    b     = A * X0(:);
    beta1 = [1e-6, 5*1e-6, 1e-5, 5*1e-5 ];
   
    for k = 1:length(beta1)
        beta = beta1(k);
         t0=tic;
        X =  tensor_cpl_admm( A, b, beta, alpha, nway, maxItr, myNorm, 0 );
        X = reshape( X, nway );
        time=toc(t0);
        X = min( 255, max( X, 0 ));
        
        for i=1:1:3
            PSNRvector(i)=psnr3(X0(:,:,i)/255,X(:,:,i)/255);
        end
        psnr = mean(PSNRvector);
        
        for i=1:1:3
            SSIMvector(i)=ssim3(X0(:,:,i),X(:,:,i));
        end
        ssim = mean(SSIMvector);
        
        display(sprintf('psnr=%.2f,ssim=%.4f,beta=%.6f', psnr, ssim, beta))
        display(sprintf('=================================='))
        
        imname=[num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_beta_',num2str(beta,'%.6f'),'.mat'];
        save(imname,'X','time');
    end
end

%% TMac-TT
j = j+1;
if EN_TMacTT
     %%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);    
    opts = [ ];
    opts.alpha = weightTC(Nway);
    opts.tol   = 1e-4;
    opts.maxit = 1000;
    opts.Xtrue = Xtrue;
    th = [0.01 0.02 0.03];  
    for k = 1:length(th)
        opts.th = th(k);
        t0=tic;
        [X, Out] = TMac_TT( Xkn, Known, opts );
        X = vdt2image256(X);
        X = min( 255, max( X, 0 ));
        time=toc(t0);
        
        for i=1:1:3
            PSNRvector(i)=psnr3(X0(:,:,i)/255,X(:,:,i)/255);
        end
        psnr = mean(PSNRvector);
        
        for i=1:1:3
            SSIMvector(i)=ssim3(X0(:,:,i),X(:,:,i));
        end
        ssim = mean(SSIMvector);
                
          display(sprintf('psnr=%.2f,ssim=%.4f,th=%.2f',psnr, ssim, opts.th))
          display(sprintf('=================================='))
                
          imname=[num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_th_',num2str(opts.th),'.mat'];
          save(imname,'X','time');
    end
end
%% KBR
j = j+1;
if EN_KBR
    %%%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);
    
    Omega     = zeros( nway );
    Omega(Known) = 1;
    Omega     = (Omega > 0);
    
    opts = [];   
    opts.tol    = 1e-4;
    opts.maxit  = 1000;
    opts.rho    = 1.1;
%     opts.lambda = 0.01; % 0.01 is even better
%     opts.mu     = 5*1e-5;  
    
    for lambda = [0.001 0.01 0.05 0.1 0.5 1 5 10]    
      for mu =  [1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 5*1e-5, 1e-3]
          opts.lambda = lambda;
           opts.mu = mu;
    
    t0=tic;
    X = KBR_TC( X0.*Omega, Omega, opts );
    time=toc(t0);      
    X    = min( 255, max( X, 0 ));
        
    for i=1:1:3
            PSNRvector(i)=psnr3(X0(:,:,i)/255,X(:,:,i)/255);    
    end
    psnr = mean(PSNRvector);
        
       
    for i=1:1:3
            SSIMvector(i)=ssim3(X0(:,:,i),X(:,:,i));    
    end
    ssim = mean(SSIMvector);
    display(sprintf('psnr=%.2f,ssim=%.4f,lambda=%.2f,mu=%.5f', psnr, ssim, opts.lambda, opts.mu))
    display(sprintf('=================================='))
        
    imname = [num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_lambda_',num2str(opts.lambda),'_mu_',num2str(opts.mu),'.mat'];
    save(imname,'X','time'); 
      end
     end
end

%% TT-TV
j = j+1;
if EN_TT_TV 
    %%%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);
    
    Nway_TV = [4 4 4 4 4 4 4 4 3]; 
    P_TV = round(SR*prod(Nway_TV));      % prod返回乘积
    Known_TV = randsample(prod(Nway_TV),P_TV);
    [Known_TV,~] = sort(Known_TV);
     X_TV = CastImageAsKet22( X0, Nway_TV, 2, 2);    
    Xkn_TV          = X_TV(Known_TV);
    Xmiss_TV        = zeros(Nway_TV);
    Xmiss_TV(Known_TV) = Xkn_TV;
    Xmiss_TV = CastKet2Image22(Xmiss_TV,256,256,2,2);
    
    
    opts = [];
    opts.X0 = X0;
    opts.alpha  = weightTC(Nway);
    opts.tol    = 1e-4;
    opts.maxit  = 200;
    opts.rho    = 10^(-3);
   opts.beta1 = 5*10^(-3); opts.beta2  = 0.1;  
    
    th     = [0.01];
    lambda = [0.3];%0.01 0.03 0.05 0.1 0.3
    beta3  = [0.3];%0.01 0.03 0.05 0.1 0.3
    for m = 1:length(th)
        opts.th = th(m);
    for k = 1:length(lambda)
        opts.lambda = lambda(k);
    for l = 1:length(beta3)
        opts.beta3 = beta3(l);
                t0=tic;
                [X, Out_TT_TV] = TT_TV( Xkn_TV, Known_TV, Nway_TV, opts );
                X    = CastKet2Image22(X,256,256,2,2);
               time=toc(t0);  
                for i=1:1:3
                    PSNRvector(i) = psnr3(X0(:,:,i)/255,X(:,:,i)/255);
                end
                psnr = mean(PSNRvector);
                                                     
                for i=1:1:3
                     SSIMvector(i)=ssim3(X0(:,:,i),X(:,:,i));
                end
                ssim = mean(SSIMvector);
                
                display(sprintf('psnr=%.2f,ssim=%.4f,th=%.2f',psnr, ssim, opts.th))
                display(sprintf('=================================='))
                
                imname=[num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_th_',num2str(opts.th),'_lambda_',num2str(opts.lambda),'_beta3_',num2str(opts.beta3),'.mat'];
                save(imname,'X','time');
    end
    end
    end
end    
%% GLON
j = j+1;
if EN_GLON 
    %%%%%
    fprintf('\n');
    disp(['performing ',methodname{j}, ' ... ']);
    
    opts = [];
    opts.tol    = 2*1e-3;
    opts.maxit  = 100;
    opts.X0     = X0;
    opts.rho    = 5;
    
    for th = [0.01]
        for sigma1 =[0.04]
            for sigma2 =[100]
                for beta1 = [100]
                    for beta2 =[20]
                        opts.th     = th;
                        opts.sigma1  = sigma1;
                        opts.sigma2  = sigma2;
                        opts.beta1   = beta1;
                        opts.beta2   = beta2;      
                         t0=tic;   
                        [X, Out_TT_FFDnet] = TT_FFDnet_BM3D( Xkn, Known, Nway, opts);
                       
                        X = vdt2image256(X);
                        X = min( 255, max( X, 0 ));
                         time=toc(t0);  
                        for i=1:1:3
                            PSNRvector(i) = psnr3(X0(:,:,i)/255,X(:,:,i)/255);
                        end
                        psnr = mean(PSNRvector);
                                                     
                        for i=1:1:3
                            SSIMvector(i)=ssim3(X0(:,:,i),X(:,:,i));
                        end
                        ssim = mean(SSIMvector);
                                       
                        display(sprintf('psnr=%.2f,ssim=%.4f,th=%.2f,sigma1=%.6f,sigma2=%.6f,beta1=%.8f,beta2=%.8f',psnr, ssim, opts.th, opts.sigma1, opts.sigma2, opts.beta1, opts.beta2))
                        display(sprintf('=================================='))
                            
                        imname=[num2str(name{1}),'_SR_',num2str(SR),'_result_',num2str(methodname{j}),'_psnr_',num2str(psnr,'%.2f'),'_ssim_',num2str(ssim,'%.4f'),'_th_',num2str(opts.th),'_sigma1_',num2str(opts.sigma1),'_sigma2_',num2str(opts.sigma2),'_beta1_',num2str(opts.beta1),'_beta2_',num2str(opts.beta2),'.mat'];
                        save(imname,'X','Out_TT_FFDnet','time');
                    end
                end
            end
        end
    end
end    
end