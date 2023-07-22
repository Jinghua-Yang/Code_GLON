function [Y_tensor, A, X, Out]= inc_SMF_LRTC(Y_tensorP, known, opts, opts2, opts5)

maxit    = opts.maxit;  
tol      = opts.tol;       
alpha    = opts.alpha;     
rho1     = opts.rho1;       
rho2     = opts.rho2;      
rho3     = opts.rho3;      
rhoX     = opts2.rho;      
rhoA     = opts5.rho;       
max_rank = opts.max_rank; 
R        = opts.R;
rank_inc = 5;

Out.Res=[];

%% Initiation
Y_tensor0 = Y_tensorP;
Nway      = size(Y_tensorP);
coNway    = zeros(1,3);
Y0        = cell(1,3);
A0        = cell(1,3);
X0        = cell(1,3);
for n = 1:3
    coNway(n) = prod(Nway)/Nway(n);
end
for i = 1:3
    Y0{i} = Unfold(Y_tensor0,Nway,i);
    Y0{i} = Y0{i}';
    X0{i} = rand(coNway(i), R(i));
    A0{i} = rand(R(i),Nway(i));
end

Y_p=Y0;X_p=X0;A_p=A0;


%% Initiation for rank increasing scheme
YY1=Fold(A0{1}'*X0{1}',Nway,1);   YY1(known)=Y_tensor0(known);
YY2=Fold(A0{2}'*X0{2}',Nway,2);   YY2(known)=Y_tensor0(known);
YY3=Fold(A0{3}'*X0{3}',Nway,3);   YY3(known)=Y_tensor0(known);
resrank1=norm(Y_tensor0(:)-YY1(:));
resrank2=norm(Y_tensor0(:)-YY2(:));
resrank3=norm(Y_tensor0(:)-YY3(:));

%% 
for k=1: maxit

    %% update X
    temp=A_p{1}*A_p{1}'+rho1*(eye(size(A_p{1}*A_p{1}')));
    X{1}=(Y_p{1}*A_p{1}'+rho1*X_p{1})*pinv(temp);

    temp=A_p{2}*A_p{2}'+rho2*(eye(size(A_p{2}*A_p{2}')));
    X{2}=(Y_p{2}*A_p{2}'+rho2*X_p{2})*pinv(temp);
    
    x_size=Nway(1:2);
    XX=Framelet_X3(A_p{3}',Y_p{3}',X_p{3}',rhoX,opts2,x_size);
    X{3}=XX';

    %% update A
    temp=X{1}'*X{1}+rho1*eye(size(X{1}'*X{1}));
    A{1}= pinv(temp)*(X{1}'*Y_p{1}+rho1*A_p{1});
    
    temp=X{2}'*X{2}+rho2*eye(size(X{2}'*X{2}));
    A{2}= pinv(temp)*(X{2}'*Y_p{2}+rho2*A_p{2});
    
    AA=TV_A3(A_p{3}',Y_p{3}',X{3}',rhoA,opts5);
    A{3}=AA';
    
    %% update Y 
    Y{1} = (X{1}*A{1}+rho3*Y_p{1})/(1+rho1); Y1 = Fold(Y{1}', Nway, 1); 
    Y{2} = (X{2}*A{2}+rho3*Y_p{2})/(1+rho2); Y2 = Fold(Y{2}', Nway, 2); 
    Y{3} = (X{3}*A{3}+rho3*Y_p{3})/(1+rho3); Y3 = Fold(Y{3}', Nway, 3);
    
    Y_tensor = alpha(1)*Y1+alpha(2)*Y2+alpha(3)*Y3;
    Y_tensor(known) = Y_tensor0(known);
    
    Res = norm(Y_tensor(:)-Y_tensorP(:))/norm(Y_tensorP(:));
    Out.Res = [Out.Res,Res];

    Y_tensorP = Y_tensor;
    
    %% update Rank  
    resold1=resrank1;
    resold2=resrank2;
    resold3=resrank3;
    
    YY1=Fold(A{1}'*X{1}',Nway,1);   YY1(known)=Y_tensor0(known);
    YY2=Fold(A{2}'*X{2}',Nway,2);   YY2(known)=Y_tensor0(known);
    YY3=Fold(A{3}'*X{3}',Nway,3);   YY3(known)=Y_tensor0(known);
    
    resrank1=norm(Y_tensor0(:)-YY1(:));
    resrank2=norm(Y_tensor0(:)-YY2(:));
    resrank3=norm(Y_tensor0(:)-YY3(:));
    
    ifrank1 = abs(1-resrank1/resold1);
    ifrank2 = abs(1-resrank2/resold2);
    ifrank3 = abs(1-resrank3/resold3);
    
    nowrank=[size(A{1},1),size(A{2},1),size(A{3},1)];
    
    fprintf('Iteration: %i     ',k);
    fprintf('nowrank:');
    fprintf(' %.0f',nowrank);
    fprintf('     ');
    fprintf('RelCha: %.6f     ',Res);
    fprintf('\n');
    
    if ifrank1<0.01 && nowrank(1)<max_rank(1)
    [A{1},X{1}]=rank_inc_adaptive(A{1},X{1},rank_inc);
    end
    
    if ifrank2<0.01 && nowrank(2)<max_rank(2)
    [A{2},X{2}]=rank_inc_adaptive(A{2},X{2},rank_inc);
    end
    
    if ifrank3<0.01 && nowrank(3)<max_rank(3)
    [A{3},X{3}]=rank_inc_adaptive(A{3},X{3},rank_inc);
    end

    

    %% check stopping criterion
    if Res<tol
        break
    end
    
    Y{1} = Unfold(Y_tensor,Nway,1); Y{1} = Y{1}';
    Y{2} = Unfold(Y_tensor,Nway,2); Y{2} = Y{2}';
    Y{3} = Unfold(Y_tensor,Nway,3); Y{3} = Y{3}';
    
    Y_p=Y;X_p=X;A_p=A;    
        
end
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