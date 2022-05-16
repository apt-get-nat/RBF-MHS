function w = RBF_FD_PHS_pol_weights_3D (x,y,z,m,d)

% Input parameters
% x,y,z     Column vectors with node locations for a single stencil; 
%           approximation to be accurate at x(1),y(1),z(1) (typically
%           the stencil 'center')
%     m     Power of r in RBF fi(r) = r^m, with m odd, >= 3.
%     d     Degree of supplementary polynomials (if d = -1 no polynomials)
%  
% Output parameter 
%     w     Matrix with nine columns, containing RBF-FD weights for 
%           d/dx, d/dy, d/dz and for d2/{dx2,dxy,dy2,dxz,dyz,dz2}, returned
%           in w(:,1), w(:,2), ... , w(:,9), respectively. respectively.

x = x-x(1); y = y-y(1); z = z-z(1); % Shift so stencil centered at origin 
n  = length(x);     % Extract the number of nodes in the stencil
np = 0;             % np will hold the number of polynomial terms

% ------ RBF part --------------------------------------------------------
dx = bsxfun(@minus,x,x');dy = bsxfun(@minus,y,y');dz = bsxfun(@minus,z,z');
r2 = dx.^2+dy.^2+dz.^2;
r  = sqrt(r2);
A0 = r.*r2.^((m-1)/2);                      % A-matrix

r  = r(:,1);                                % Get first column of r
rm4 = r.^(m-4);  rm2 = r.*r.*rm4;           % Right hand sides
L0 = m*[ -x.*rm2, ...                       % d/dx
         -y.*rm2, ...                       % d/dy
         -z.*rm2, ...                       % d/dz
         ((m-1)*x.^2+y.^2+z.^2).*rm4, ...   % d2/dx2
         (m-2)*x.*y.*rm4, ...               % d2/dxdy
         (x.^2+(m-1)*y.^2+z.^2).*rm4, ...   % d2/dy2    
         (m-2)*x.*z.*rm4, ...               % d2/dxdz
         (m-2)*y.*z.*rm4, ...               % d2/dydz
         (x.^2+y.^2+(m-1)*z.^2).*rm4];      % d2/dz2

% ------ Polynomial part -------------------------------------------------
if d == -1                            % Special case; no polynomial terms,
    A = A0;  L = L0;                  % i.e. pure RBF    
else    % Create matrix with polynomial terms and matching constraints
    X   =  x(:,ones(1,d+1));  X(:,1) = 1;  X = cumprod(X,2);
    Y   =  y(:,ones(1,d+1));  Y(:,1) = 1;  Y = cumprod(Y,2);
    Z   =  z(:,ones(1,d+1));  Z(:,1) = 1;  Z = cumprod(Z,2);
    np  = (d+3)*(d+2)*(d+1)/6;        % Number of polynomial terms
    XYZ = zeros(n,np); col = 0;       % Assemble polynomial matrix block
    for kt = 0:d
        for kz = 0:kt
            for ky = 0:kt-kz
                col = col+1;
                kx = kt-ky-kz;
                XYZ(:,col) = X(:,kx+1).*Y(:,ky+1).*Z(:,kz+1);
             end
        end
    end
    L1 = zeros(np,9);                 % Create matching RHSs    
    if d >= 1; L1(2,1) = 1; L1(3,2) = 1; L1( 4,3) = 1; end
    if d >= 2; L1(5,4) = 2; L1(6,5) = 1; L1( 7,6) = 2; 
               L1(8,7) = 1; L1(9,8) = 1; L1(10,9) = 2; end    
    A = [A0,XYZ;XYZ',zeros(col,col)]; % Assemble linear system to be solved
    L = [L0;L1];                      % Assemble RHSs   
end

% ------ Solve for weights -----------------------------------------------
W = A\L;
w = W(1:n,:);                 % Extract the RBF-FD weights