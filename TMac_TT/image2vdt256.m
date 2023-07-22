function [B] = image2vdt256(B)

nway = [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3];
B = reshape(B, nway);
per_nway = [1 9 2 10 3 11 4 12 5 13 6 14 7 15 8 16 17];
B = permute(B, per_nway);
B = reshape(B, [4 4 4 4 4 4 4 4 3]);

end