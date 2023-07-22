function [M, Out_Si_TT] = TT_TV( data, known, Nway, opts )

    Orig = opts.X0;
    alpha = opts.alpha; th = opts.th; maxiter = opts.maxit; tol = opts.tol;
    lambda = opts.lambda; beta1 = opts.beta1; beta2 = opts.beta2; beta3 = opts.beta3; rho = opts.rho;
    %% Initialization
    R = 256; C = 256; I1 = 2; J1 = 2;
    N = length(Nway);
    M = initialization_M(Nway,known,data); 
%     [~,ranktube] = SVD_MPS_Rank_Estimation(Orig,th);   % Initial TT ranks
%     [X,Y] = initialMatrix(N,Nway,ranktube);
    
     switch th
        case 0.01
            load X_TT_01.mat
            load Y_TT_01.mat
        case 0.02
            load X_TT_02.mat
            load Y_TT_02.mat
        case 0.03
            load X_TT_03.mat
            load Y_TT_03.mat
    end
  
    dimL = zeros(1,N-1);
    dimR = zeros(1,N-1);
    IL = 1;
    for k = 1:N-1
        dimL(k) = IL*Nway(k);
        dimR(k) = prod(Nway)/dimL(k);
        IL = dimL(k);
    end 
    
    X0 = X; Y0 = Y;  M0 = M;  
    X = cell(1,N-1); Y = cell(1,N-1);
    
    N = length(Nway);
    k = 1;
    relerr = [];
    relerr(1) = 1;
       
%     close all;
%     % Initialize figures
%     h1 = figure();clf;
%     subplot(1,3,1);cla;hold on;
%     subplot(1,3,2);
%     set(h1,'Position',[200 600 1500 350]);
%     M_img = CastKet2Image22(M,R,C,I1,J1);
%     imagesc(uint8(M_img));drawnow;
%     cla;hold on;
%     subplot(1,3,3);
    fprintf('iter: RSE  \n');   
    %% Start Time measure
    t0=tic;
    while relerr(k) > tol
        k = k+1;
        Mlast = M;
        %% update (X,Y)
        for n = 1:N-1
            M_Temp = reshape(M0,[dimL(n) dimR(n)]);
            X{n}   = (alpha(n)*M_Temp*Y0{n}' + rho*X0{n})*pinv( alpha(n)*Y0{n}*Y0{n}' + rho*eye(size(Y0{n}*Y0{n}')));
            Y{n}   = pinv(alpha(n)*X{n}'*X{n} + rho*eye(size(X{n}'*X{n})))*(alpha(n)*X{n}'*M_Temp +rho*Y0{n});
        end
        
        %% update M by ADMM
        [M] = update_M(data,known,Nway,N,X,Y,M0,alpha,lambda,beta1,beta2,beta3,rho);
        
        % Calculate relative error
        relerr(k) = abs(norm(M(:)-Mlast(:)) / norm(Mlast(:)));

       %% Update figures
%         set(0,'CurrentFigure',h1);
%         subplot(1,3,1);cla;hold on;   
%         plot(relerr);
%         plot(tol*ones(1,length(relerr)));
%         grid on
%         set(gca,'YScale','log')
%         title('Relative Error')
%         ylim([(tol-5e-5) 1]);
%         subplot(1,3,2);
        M_img = CastKet2Image22(M,R,C,I1,J1);
%         imagesc(uint8(M_img));
%         title('TT-TV');        
%         drawnow; 
%         % message log
        Img0=Orig;
%         subplot(1,3,3);
%         imagesc(uint8(Img0));
%         title('Original image');
        if k == 2 || mod(k-1,20) ==0 
           fprintf('  %d:  %f \n',k-1,RSE(M_img(:),Img0(:)));
        end

        X0=X; Y0=Y; M0=M;
        %% check stopping criterion
        if k > maxiter || (relerr(k)-relerr(k-1) > 0)
%      if k > maxiter || relerr(k)<tol
            break
        end
    end
    %% Stop Time measure
    time = toc(t0);
    Out_Si_TT.time = time;
    Out_Si_TT.relerr = relerr;
end
function [X0,Y0] = initialMatrix(N,Nway,ranktube)
    X0 = cell(1,N-1);Y0 = cell(1,N-1);
    dimL = zeros(1,N-1);
    dimR = zeros(1,N-1);
    IL = 1;
    for k = 1:N-1
        dimL(k) = IL*Nway(k);
        dimR(k) = prod(Nway)/dimL(k);
        %
        X0{k} = randn(dimL(k),ranktube(k));
        Y0{k} = randn(ranktube(k),dimR(k));
        %uniform distribution on the unit sphere
        X0{k} = bsxfun(@rdivide,X0{k},sqrt(sum(X0{k}.^2,1)));
        Y0{k} = bsxfun(@rdivide,Y0{k},sqrt(sum(Y0{k}.^2,2)));
        %
        IL = dimL(k);
    end   
end