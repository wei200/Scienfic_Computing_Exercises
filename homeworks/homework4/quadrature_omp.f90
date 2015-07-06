
module quadrature_omp
    
    use omp_lib

contains

real(kind=8) function trapezoid(f,a,b,n)
    ! This funciton returns the estimate of the integral

    implicit none
    real(kind=8), intent(in)::a,b
    real(kind=8),external::f
    integer, intent(in)::n

    integer :: j
    real(kind=8):: h,sumf,xj

    h = (b - a)/(n - 1)
    sumf =  0.5d0*(f(a)+f(b))

    !$omp parallel do private(xj) reduction(+:sumf)
    do j = 2,n-1
        xj = a + (j-1)*h
        sumf = sumf + f(xj)
        enddo

    trapezoid = h*sumf

end function trapezoid

subroutine error_table(f,a,b,nvals,int_true)
    implicit none   
    integer,dimension(:),intent(in) :: nvals
    real(kind=8),external::f
    real(kind=8),intent(in)::a,b,int_true
    real(kind=8)::sum,int_trap,ratio, error, last_error
    
    integer::j,n
    
    print *, "    n       trapezoid      error        ratio"
    
    last_error = 0.d0
    do j = 1,size(nvals)
        n = nvals(j)
        int_trap = trapezoid(f,a,b,n)
        error = abs(int_trap - int_true)
        ratio = last_error/error
        last_error = error
        print 11, n, int_trap, error, ratio
11      format(i8, es22.14,es13.3,es13.3)
        enddo

end subroutine error_table

end module quadrature_omp

