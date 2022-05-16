function [J] = resBJ(Bp, n, Dx, Dy, Dz, Dxx, Dyy, Dzz, ...
                  index_x0,index_x1,index_y0,index_y1,index_z0,index_z1,index_gh, gamma)
% resBJ.m
% Author: Nathaniel Mathews
% Input:
%         Bp   - the magnetic field in column vector format, in the order
%                [ Bx ; By ; Bz ]
%         n    - the number of nodes
%         Dx   - the differentiation matrix with respect to x
%         Dy   - the differentiation matrix with respect to y
%         Dz   - the differentiation matrix with respect to z
%         Dxx  - the second derivative matrix with respect to x
%         Dyy  - the second derivative matrix with respect to y
%         Dzz  - the second derivative matrix with respect to z
%     index_x0 - a column vector whose entries are the row indices of Bp at
%                the lower x boundary
%     index_x1 - a column vector whose entries are the row indices of Bp at
%                the upper x boundary
%     index_y0 - a column vector whose entries are the row indices of Bp at
%                the lower y boundary
%     index_y1 - a column vector whose entries are the row indices of Bp at
%                the upper y boundary
%     index_z0 - a column vector whose entries are the row indices of Bp at
%                the lower z boundary
%     index_z1 - a column vector whose entries are the row indices of Bp at
%                the upper z boundary
%     index_gh - a column vector whose entries are the row indices of Bp at
%                the ghost nodes below the lower z boundary
%        gamma - the hyperviscocity parameter
%
% Output:
%         J    - the Jacobian matrix
    
    
    Bx = Bp(1:n); By = Bp(n+1:2*n); Bz = Bp(2*n+1:3*n);
    clear Bp;
    
    J = sparse(3*n,3*n);
    
    % Force balance in x
    J(1:n,1:n) = ...        % Bx block
        spdiags(By,0,n,n)*Dy + spdiags(Bz,0,n,n)*Dz;
    J(1:n,n+1:2*n) = ...    % By block
        spdiags(Dy*Bx-Dx*By,0,n,n)-spdiags(By,0,n,n)*Dx;
    J(1:n,2*n+1:3*n) = ...  % Bz block
        spdiags(Dz*Bx-Dx*Bz,0,n,n)-spdiags(Bz,0,n,n)*Dx;
    
    % Force balance in y
    J(n+1:2*n,1:n) = ...       % Bx block
        spdiags(Dx*By-Dy*Bx,0,n,n)-spdiags(Bx,0,n,n)*Dy;
    J(n+1:2*n,n+1:2*n) = ...   % By block
        spdiags(Bx,0,n,n)*Dx + spdiags(Bz,0,n,n)*Dz;
    J(n+1:2*n,2*n+1:3*n) = ... % Bz block
        spdiags(Dz*By-Dy*Bz,0,n,n)-spdiags(Bz,0,n,n)*Dy;
    
    % Force balance in z
    J(2*n+1:3*n,1:n) = ... % Bx block
        spdiags(Dx*Bz-Dz*Bx,0,n,n)-spdiags(Bx,0,n,n)*Dz;
    J(2*n+1:3*n,n+1:2*n) = ... % By block
        spdiags(Dy*Bz-Dz*By,0,n,n)-spdiags(By,0,n,n)*Dz;
    J(2*n+1:3*n,2*n+1:3*n) = ... % Bz block
        spdiags(Bx,0,n,n)*Dx + spdiags(By,0,n,n)*Dy;
    
    J = J/(4*pi);
    
    % hyperviscocity
    J = J - gamma*[Dxx+Dyy+Dzz, sparse(n,n), sparse(n,n); ...
                   sparse(n,n), Dxx+Dyy+Dzz, sparse(n,n); ...
                   sparse(n,n), sparse(n,n), Dxx+Dyy+Dzz];
    
% BC stuff
% There are a few different options for top and side boundaries. If these
% are altered they should also be altered in resBJ.m
    n1 = numel(index_z0); I = speye(n,n);
    J(2*n+index_z0,:) = sparse(n1,3*n); J(2*n+index_z0,2*n+index_z0) = speye(n1);
    J(2*n+index_z1,:) = sparse(n1,3*n);
    switch 2 % top boundary
        case 1% radiative (low92)
            J(2*n+index_z1,2*n+1:3*n) = Dz(index_z1,:) + I(index_z1,:);
        case 2% constant DzBz (giblow)
            J(2*n+index_z1,2*n+1:3*n) = Dz(index_z1,:);
    end
    J(index_gh,:)     = sparse(n1,3*n); J(index_gh    ,index_z0)     = speye(n1);
    J(n+index_gh,:)   = sparse(n1,3*n); J(n+index_gh  ,n+index_z0)   = speye(n1);
    J(2*n+index_gh,:) = [Dx(index_z0,:), Dy(index_z0,:), Dz(index_z0,:)];
    switch 3 % side boundaries
        case 1% Dperpperp Bperp + Bperp = 0 sides
            nbound = numel(index_x0);
                J(index_x0,:) = sparse(nbound,3*n);
                J(index_x0,1:n) = Dxx(index_x0,:) + I(index_x0,:);
            nbound = numel(index_x1);
                J(index_x1,:) = sparse(nbound,3*n);
                J(index_x1,1:n) = Dxx(index_x1,:) + I(index_x1,:);
            nbound = numel(index_y0);
                J(n+index_y0,:) = sparse(nbound,3*n);
                J(n+index_y0,n+1:2*n) = Dyy(index_y0,:) + I(index_y0,:);
            nbound = numel(index_y1);
                J(n+index_y1,:) = sparse(nbound,3*n);
                J(n+index_y1,n+1:2*n) = Dyy(index_y1,:) + I(index_y1,:);
        case 2 % no flux sides
            nbound = numel(index_x0);
                J(index_x0,:) = sparse(nbound,3*n);
                J(index_x0,1:n) = I(index_x0,:);
            nbound = numel(index_x1);
                J(index_x1,:) = sparse(nbound,3*n);
                J(index_x1,1:n) = I(index_x1,:);
            nbound = numel(index_y0);
                J(n+index_y0,:) = sparse(nbound,3*n);
                J(n+index_y0,n+1:2*n) = I(index_y0,:);
            nbound = numel(index_y1);
                J(n+index_y1,:) = sparse(nbound,3*n);
                J(n+index_y1,1:n) = I(index_y1,:);
        case 3% radial field sides
            nbound = numel(index_x0);
            J(index_x0,:) = sparse(nbound,3*n);
                J(index_x0,index_x0) = spdiags(Bx(index_x0),0,nbound,nbound);
                J(index_x0,n+index_x0) = spdiags(By(index_x0),0,nbound,nbound);
            nbound = numel(index_x1);
            J(index_x1,:) = sparse(nbound,3*n);
                J(index_x1,index_x1) = spdiags(Bx(index_x1),0,nbound,nbound);
                J(index_x1,n+index_x1) = spdiags(By(index_x1),0,nbound,nbound);
            nbound = numel(index_y0);
            J(n+index_y0,:) = sparse(nbound,3*n);
                J(n+index_y0,index_y0) = spdiags(Bx(index_y0),0,nbound,nbound);
                J(n+index_y0,n+index_y0) = spdiags(By(index_y0),0,nbound,nbound);
            nbound = numel(index_y1);
            J(n+index_y1,:) = sparse(nbound,3*n);
                J(n+index_y1,index_y1) = spdiags(Bx(index_y1),0,nbound,nbound);
                J(n+index_y1,n+index_y1) = spdiags(By(index_y1),0,nbound,nbound);
    end
    
end