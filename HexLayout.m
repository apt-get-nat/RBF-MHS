function [p] = HexLayout(nodes_per_layer)
    % Generates hex nodes in the unit square.
    % Builds nodes up from the y=0 axis, so the last layer may not align
    % perfectly with y=1. Suggested nodes_per_layer that get close:
    % 47 for n=2511,  60 for n=4106,  73 for n=6090, 80 for n=7314
    hx = 1./(nodes_per_layer-1);
    hy = sqrt(3)/2 * hx;
    layers = 1 + floor (1/hy);
    n = nodes_per_layer* floor((layers + 1 )/2) + (nodes_per_layer-1)*floor(layers/2);

    k = 0;
      for j = 1:layers
        y = hy * ( j - 1 );
        jmod = mod ( j, 2 );
            if ( jmod == 1 )
                for i = 1 : nodes_per_layer
                    x = ( i - 1 ) / ( nodes_per_layer - 1 );
                    k = k + 1;
                   if ( k <= n )
                      p(1,k) = x;
                      p(2,k) = y;
                   end
                end
            else
                for i = 1 : nodes_per_layer-1
                     x = ( 2 * i - 1 ) / ( 2 * nodes_per_layer - 2 );
                     k = k + 1;
                    if ( k <= n )
                       p(1,k) = x;
                       p(2,k) = y;
                    end
                end
            end
      end

end