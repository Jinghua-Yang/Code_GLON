function [B] = vdt2image256(B)

nway = [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3];
B = reshape(B, nway);
per_nway = [1 3 5 7 9 11 13 15 2 4 6 8 10 12 14 16 17];
B = permute(B, per_nway);
B = reshape(B, [256 256 3]);

end