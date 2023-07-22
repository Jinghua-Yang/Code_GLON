function [M, Out_Si_TT] = TT_FFDnet_BM3D( data, known, Nway, opts)
maxit = opts.maxit; 
tol   = opts.tol;
rho   = opts.rho;
alpha = weightTC(Nway);

opts.alpha = alpha;
%% Initialization
N = length(Nway);
M = initialization_M(Nway,known,data);

ranktube = [4 16 5 5 5 5 12 3];
%     [X,Y] = initialMatrix(N,Nway,ranktube);
%     save X.mat X
%     save Y.mat Y

load X.mat
load Y.mat



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
relerr2 = [];
Da = [];

max_rank = [30 50 50 30];
rank_inc = 5;

for i=1:4
    MM{i}=reshape(X0{i+2}*Y0{i+2},Nway); 
    MMi = MM{i};
    resrank{i}=norm(M0(:)- MMi(:));
end


Mimg  =  vdt2image256(M0); % 3 order
[w,h,c] = size(Mimg);
sigma2 = opts.sigma2;
beta2  = opts.beta2; 
rho    = opts.rho;
Pimg = Mimg; 

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
    
          %% update P
      input2 = (beta2*Mimg - rho*Pimg)/(beta2+rho);
       B = zeros(1,c);
       for i= 1:c
           Temp2 = input2(:,:,i);
           B(i) = max(Temp2(:));
           input2(:,:,i) =  Temp2/B(i);
       end
       
      max_in2 = max(input2(:));min_in2 = min(input2(:));
      input2 = (input2-min_in2)/(max_in2-min_in2);
      sigmas2 = sigma2/(max_in2-min_in2);
       
       [~, Pimg] = CBM3D(1, input2, sigmas2);  
       
       Pimg(Pimg<0)=0;Pimg(Pimg>1)=1;
       Pimg = Pimg*(max_in2-min_in2)+min_in2;  
       
       for i= 1:c
            Pimg(:,:,i) =   B(i)*Pimg(:,:,i);
       end
    
    
    
    %% update M by ADMM
    [M,re] = FFDnet_BM3D_M(data,known,Nway,N,X,Y,M0,opts,Pimg);
    
   
    % Calculate relative error
    relerr(k) = abs(norm(M(:)-Mlast(:)) / norm(Mlast(:)));
    relerr2(k) = abs(norm(M(:)-Mlast(:)) );
    Da=M;
    X0=X; Y0=Y; M0=M;
    %% check stopping criterion
%     if k > maxit || (relerr(k)-relerr(k-1) > 0)
    if k > maxit ||  relerr(k) < tol  
        break 
    end
    
    %%
    if k == 25
        rho = rho*1.5;%618;%618;%618;
    
    end
    
    if k == 30
        rho = rho*1.5;
      
    end
    
    if k == 35
        rho = rho*1.5;
     
    end
    
    if k > 40
        rho = rho*1.2;
       
    end
    
   beta2=beta2*1.2;
    %% update Rank  
    for i=1:4
       resold{i}=resrank{i};
       MM{i}=reshape(X{i+2}*Y{i+2},Nway); 
       MMi = MM{i};
       MMi(known)=M0(known);
       resrank{i}=norm(M0(:)-MMi(:));
       ifrank{i} = abs(1-resrank{i}/resold{i});
       nowrank=[size(X{1},1),size(X{2},1),size(X{3},1),size(X{4},2),size(X{5},2),size(X{6},2),size(X{7},2),size(X{8},2)];
       if ifrank{i}<0.01 && nowrank(i)<max_rank(i)
          [X{i+2},Y{i+2}]=rank_inc_adaptive(X{i+2},Y{i+2},rank_inc);
       end
    end
end
%% Stop Time measure
time = toc(t0);
Out_Si_TT.time = time;
Out_Si_TT.relerr = relerr;
end

function [A,X]=rank_inc_adaptive(A,X,rank_inc)
    % increase the estimated rank
    for ii = 1:rank_inc
        rdnx = rand(size(X,1),1);
        rdna = rand(1,size(A,2));
        X = [X,rdnx];
        A = [A;rdna];
    end
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