function [M] = update_M(data,known,Nway,N,X,Y,M0,alpha,lambda,beta1,beta2,beta3,rho)
    R=256; C = 256; I1 = 2; J1 = 2;
    A = cell(1,N-1); B = cell(1,N-1);
    for i = 1:N-1
        A{i} = M0;
        B{i} = zeros(size(A{i}));
    end
    M_img  =  CastKet2Image22(M0,R,C,I1,J1); % 3 order
    Z_img  =  M_img; % 3 order
    Q_img  =  zeros(size(M_img)); % 3 order
    
    [Z1.x, Z1.y] = grad(Z_img(:,:,1),'circular');
    [Z2.x, Z2.y] = grad(Z_img(:,:,2),'circular');
    [Z3.x, Z3.y] = grad(Z_img(:,:,3),'circular');
    E1.x = Z1.x; E1.y = Z1.y;
    E2.x = Z2.x; E2.y = Z2.y;
    E3.x = Z3.x; E3.y = Z3.y;
    F1.x = zeros(R,C); F1.y = zeros(R,C);
    F2.x = zeros(R,C); F2.y = zeros(R,C);
    F3.x = zeros(R,C); F3.y = zeros(R,C);
    denom = psf2otf([0,1,0;1,-4,1;0,1,0],[R,C]);
    
        for r = 1:10
            
           %% update A
            for n = 1:N-1
                A{n} = alpha(n)*reshape(X{n}*Y{n},Nway)+beta1*M0+B{n};
                A{n} = A{n}/(alpha(n)+beta1);
            end
            
           %% update M
            Z = CastImageAsKet22(Z_img,Nway,I1,J1);
            Q = CastImageAsKet22(Q_img,Nway,I1,J1);
            M = beta2*Z-Q+rho*M0;
            for n = 1:N-1
                M = M+beta1*A{n}-B{n};
            end
            M = M./((N-1)*beta1+beta2+rho);
%             M = CastKet2Image22(M,R,C,I1,J1);
            M(known) = data;
%             M = CastImageAsKet22( M, [4 4 4 4 4 4 4 4 3], I1, J1);
            M_img = CastKet2Image22(M,R,C,I1,J1);
                      
           %% update Z
            temp1 = -div(beta3*E1.x-F1.x,beta3*E1.y-F1.y);
            temp1 = temp1 + beta2*M_img(:,:,1) + Q_img(:,:,1);
            Z_img(:,:,1) = real( ifft2( fft2(temp1)./(beta2-beta3*denom) ) );
        
            temp2 = -div(beta3*E2.x-F2.x,beta3*E2.y-F2.y);
            temp2 = temp2 + beta2*M_img(:,:,2) + Q_img(:,:,2);
            Z_img(:,:,2) = real( ifft2( fft2(temp2)./(beta2-beta3*denom) ) );
        
            temp3 = -div(beta3*E3.x-F3.x,beta3*E3.y-F3.y);
            temp3 = temp3 + beta2*M_img(:,:,3) + Q_img(:,:,3);
            Z_img(:,:,3) = real( ifft2( fft2(temp3)./(beta2-beta3*denom) ) );
        
           %% update E
            [Z1.x, Z1.y] = grad(Z_img(:,:,1),'circular');
            w1x    = Z1.x + F1.x/beta3;
            w1y    = Z1.y + F1.y/beta3;
            absw1  = sqrt(w1x.^2+w1y.^2); 
            w1temp = max(0,absw1-lambda/beta3);
            absw1(absw1==0)=1;
            E1.x = w1temp.*w1x./absw1;
            E1.y = w1temp.*w1y./absw1;
        
            [Z2.x, Z2.y] = grad(Z_img(:,:,2),'circular');
            w2x    = Z2.x + F2.x/beta3;
            w2y    = Z2.y + F2.y/beta3;
            absw2  = sqrt(w2x.^2+w2y.^2); 
            w2temp = max(0,absw2-lambda/beta3);
            absw2(absw2==0)=1;
            E2.x = w2temp.*w2x./absw2;
            E2.y = w2temp.*w2y./absw2;
        
            [Z3.x, Z3.y] = grad(Z_img(:,:,3),'circular');
            w3x    = Z3.x + F3.x/beta3;
            w3y    = Z3.y + F3.y/beta3;
            absw3  = sqrt(w3x.^2+w3y.^2); 
            w3temp = max(0,absw3-lambda/beta3);
            absw3(absw3==0)=1;
            E3.x = w3temp.*w3x./absw3;
            E3.y = w3temp.*w3y./absw3;
        
          %% update B,Q,F
           for n =1:N-1
               B{n} = B{n}+beta1*(M - A{n});
           end
           Q_img = Q_img + beta2*(M_img - Z_img);
           F1.x = F1.x + beta3*(Z1.x-E1.x); F1.y = F1.y + beta3*(Z1.y-E1.y);
           F2.x = F2.x + beta3*(Z2.x-E2.x); F2.y = F2.y + beta3*(Z2.y-E2.y);
           F3.x = F3.x + beta3*(Z3.x-E3.x); F3.y = F3.y + beta3*(Z3.y-E3.y);            
        end
end