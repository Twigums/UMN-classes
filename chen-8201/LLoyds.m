X = csvread("pr1.csv");
[c1, c2, c11, c12] = LLoyds(X, [0, 0], [2, -0.5], 1);
[idx,C] = kmeans(X,2);
figure;
plot(c11(:,1),c11(:, 2),'r.','MarkerSize',12);
hold on
plot(c12(:,1),c12(:, 2),'b.','MarkerSize',12);
hold off
saveas(gcf, "Lloyds2.png")

%figure;
%plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12);
%hold on
%plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12);
%hold off

function [c1,c2,cl1,cl2] = LLoyds(T,g1,g2,iter)
%T = data input, m data points x n variables array (this case 200x2)
%iter = number of iterations
%g1 = coordinates of guess point 1 [x1 y1]
%g2 = coordinates of guess point 2 [x2 y2]

for j=1:iter
    class1=[];
    class2=[];
    for jj=1:length(T)
        d1=norm(g1-T(jj,:));
        d2=norm(g2-T(jj,:));
        if d1<d2
            class1=[class1; [T(jj,1) T(jj,2)]];
        else
            class2=[class2; [T(jj,1) T(jj,2)]];
        end
    end
    g1=[mean(class1(1:end,1)) mean(class1(1:end,2))];
    g2=[mean(class2(1:end,1)) mean(class2(1:end,2))];
end
c1=g1;
c2=g2;
cl1=class1;
cl2=class2;
end