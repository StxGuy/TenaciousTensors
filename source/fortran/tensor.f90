module TenaciousTensor
    implicit none

    integer,parameter,private   :: dp = kind(1.d0)        
    
    type Tensor
        integer                 :: rank
        real(dp),allocatable    :: rank2(:,:)
        real(dp),allocatable    :: rank3(:,:,:)
        real(dp),allocatable    :: rank4(:,:,:,:)
        
        contains
        
        procedure           :: shadow
        procedure           :: randomize
        procedure           :: zero
        
        ! Gradient procedures
        procedure           :: GradDesc
        !procedure           :: GradLoss
        
        final               :: tensor_destructor
        
    end type 
    
    interface Tensor
        procedure           :: tensor_constructor_2
        procedure           :: tensor_constructor_3
        procedure           :: tensor_constructor_4
    end interface
    
    interface operator(**)
        module procedure prod_tensor
    end interface
    
    interface operator(*)
        module procedure Had_tensor
        module procedure real_prod
    end interface
    
    interface operator(+)
        module procedure sum_tensor
    end interface
    
    interface operator(-)
        module procedure sub_tensor
    end interface
    
    interface assignment(=)
        module procedure equal_tensor_1
        module procedure equal_tensor_2
        module procedure equal_tensor_3
        module procedure equal_tensor_4
    end interface
    
    ! Intrinsic functions
    interface transpose
        module procedure tensor_transp
    end interface
    
    contains
    
    !==============================================================!
    !                  CONSTRUCTORS & DESTRUCTORS                  !
    !==============================================================!
    ! rank-2
    function tensor_constructor_2(a,b) result(self)
        implicit none
        
        integer,intent(in)  :: a,b
        type(Tensor)        :: self
        
        self%rank = 2
        allocate(self%rank2(a,b))
        
        self%rank2 = reshape([1,2,3,4],shape(self%rank2))
    end function
    
    ! rank-3
    function tensor_constructor_3(a,b,c) result(self)
        implicit none
        
        integer,intent(in)  :: a,b,c
        type(Tensor)        :: self
        
        self%rank = 3
        allocate(self%rank3(a,b,c))
    end function
    
    ! rank-4
    function tensor_constructor_4(a,b,c,d) result(self)
        implicit none
        
        integer,intent(in)  :: a,b,c,d
        type(Tensor)        :: self
        
        self%rank = 4
        allocate(self%rank4(a,b,c,d))
    end function
        
    
    ! Destructor
    subroutine tensor_destructor(self)
        implicit none
        
        type(Tensor)    :: self
        
        if (allocated(self%rank2)) then
            deallocate(self%rank2)
        end if
        if (allocated(self%rank3)) then
            deallocate(self%rank3)
        end if
        if (allocated(self%rank4)) then
            deallocate(self%rank4)
        end if
    end subroutine
    
    !==============================================================!
    !                          OPERATORS                           !
    !==============================================================!
    
    ! (%)
    function Prod_tensor(a,b) result(c)
        implicit none
        
        type(Tensor),intent(in) :: a,b
        type(Tensor)            :: c
        
        select case(a%rank)
            case (2)
                c = Tensor(size(a%rank2,1),size(b%rank2,2))
                c%rank2 = matmul(a%rank2,b%rank2)
        end select
    end function
    
    ! (*)
    function Had_tensor(a,b) result(c)
        implicit none
        
        type(Tensor),intent(in) :: a,b
        type(Tensor)            :: c
        
        select case(a%rank)
            case (2)
                c = Tensor(size(a%rank2,1),size(a%rank2,2))
                c%rank2 = a%rank2 * b%rank2
            case (3)
                c = Tensor(size(a%rank3,1),size(a%rank3,2),size(a%rank3,3))
                c%rank3 = a%rank3 * b%rank3
            case (4)
                c = Tensor(size(a%rank4,1),size(a%rank4,2),size(a%rank4,3),size(a%rank4,4))
                c%rank4 = a%rank4 * b%rank4
        end select
    end function
    
    ! real x tensor
    function real_prod(a,b) result(c)
        implicit none
        
        real(dp),intent(in)         :: a
        class(Tensor),intent(in)    :: b
        type(Tensor)                :: c
        
        select case(b%rank)
            case (2)
                c = Tensor(size(b%rank2,1),size(b%rank2,2))
                c%rank2 = a*b%rank2
            case (3)
                c = Tensor(size(b%rank3,1),size(b%rank3,2),size(b%rank3,3))
                c%rank3 = a*b%rank3
            case (4)
                c = Tensor(size(b%rank4,1),size(b%rank4,2),size(b%rank4,3),size(b%rank4,4))
                c%rank4 = a*b%rank4
        end select
    end function
    
        
    ! (+)
    function sum_tensor(a,b) result(c)
        implicit none
        
        type(Tensor),intent(in) :: a,b
        type(Tensor)            :: c
        
        select case(a%rank)
            case (2)
                c = Tensor(size(a%rank2,1),size(a%rank2,2))
                c%rank2 = a%rank2 + b%rank2
            case (3)
                c = Tensor(size(a%rank3,1),size(a%rank3,2),size(a%rank3,3))
                c%rank3 = a%rank3 + b%rank3
                
            case (4)
                c = Tensor(size(a%rank4,1),size(a%rank4,2),size(a%rank4,3),size(a%rank4,4))
                c%rank4 = a%rank4 + b%rank4
        end select
    end function

    ! (-)
    function sub_tensor(a,b) result(c)
        implicit none
        
        type(Tensor),intent(in) :: a,b
        type(Tensor)            :: c
        
        select case(a%rank)
            case (2)
                c = Tensor(size(a%rank2,1),size(a%rank2,2))
                c%rank2 = a%rank2 - b%rank2
            case (3)
                c = Tensor(size(a%rank3,1),size(a%rank3,2),size(a%rank3,3))
                c%rank3 = a%rank3 - b%rank3
                
            case (4)
                c = Tensor(size(a%rank4,1),size(a%rank4,2),size(a%rank4,3),size(a%rank4,4))
                c%rank4 = a%rank4 - b%rank4
        end select
    end function    
    
    ! (=)
    subroutine equal_tensor_1(new,name)
        type(Tensor),intent(out)    :: new
        type(Tensor),intent(in)     :: name
        
        select case(name%rank)
            case (2)
                new%rank = 2
                allocate(new%rank2(size(name%rank2,1),size(name%rank2,2)))
                new%rank2 = name%rank2
            case (3)
                new%rank = 3
                allocate(new%rank3(size(name%rank3,1),size(name%rank3,2),size(name%rank3,3)))
                new%rank3 = name%rank3
            case (4)
                new%rank = 4
                allocate(new%rank4(size(name%rank4,1),size(name%rank4,2),size(name%rank4,3),size(name%rank4,4)))
                new%rank4 = name%rank4
        end select
    end subroutine

    ! (=) rank2
    subroutine equal_tensor_2(new,name)
        type(Tensor),intent(out)    :: new
        real,intent(in)             :: name(:,:)
        
        new%rank = 2
        allocate(new%rank2(size(name,1),size(name,2)))
        new%rank2 = name
    end subroutine
    
    ! (=) rank3
    subroutine equal_tensor_3(new,name)
        type(Tensor),intent(out)    :: new
        real,intent(in)             :: name(:,:,:)
        
        new%rank = 3
        allocate(new%rank3(size(name,1),size(name,2),size(name,3)))
        new%rank3 = name
    end subroutine
    
    ! (=) rank4
    subroutine equal_tensor_4(new,name)
        type(Tensor),intent(out)    :: new
        real,intent(in)             :: name(:,:,:,:)
        
        new%rank = 4
        allocate(new%rank4(size(name,1),size(name,2),size(name,3),size(name,4)))
        new%rank4 = name
    end subroutine
    
       
        

    !==============================================================!
    !                         PROCEDURES                           !
    !==============================================================!
    ! Create a new tensor with the same rank as another
    function shadow(self) result(X)
        implicit none
        
        class(Tensor),intent(in)    :: self
        type(Tensor)                :: X
        
        select case(self%rank)
            case (2)
                X = Tensor(size(self%rank2,1),size(self%rank2,2))
            case (3)
                X = Tensor(size(self%rank3,1),size(self%rank3,2),size(self%rank3,3))
            case (4)
                X = Tensor(size(self%rank4,1),size(self%rank4,2),size(self%rank4,3),size(self%rank4,4))
        end select
    end function
    
    ! Randomize a tensor
    subroutine randomize(self)
        implicit none
        
        class(Tensor),intent(inout) :: self
        
        select case(self%rank)
            case(2)
                call random_number(self%rank2)
                self%rank2 = 2*(self%rank2 - 0.5)
            case(3)
                call random_number(self%rank3)
                self%rank3 = 2*(self%rank3 - 0.5)
            case(4)
                call random_number(self%rank4)
                self%rank4 = 2*(self%rank4 - 0.5)
        end select
    end subroutine
    
    ! Zero a tensor
    subroutine zero(self)
        implicit none
        
        class(Tensor),intent(inout) :: self
        
        select case(self%rank)
            case(2)
                self%rank2 = 0.0
            case(3)
                self%rank3 = 0.0
            case(4)
                self%rank4 = 0.0
        end select
    end subroutine
    
    ! Transpose a tensor
    function tensor_transp(X) result(Y)
        implicit none
        
        class(Tensor),intent(in)    :: X
        type(Tensor)                :: Y
        
        select case(X%rank)
            case(2)
                Y = Tensor(size(X%rank2,2),size(X%rank2,1))
                Y%rank2 = transpose(X%rank2)
        end select
    end function
        
    
    
    !---------------------------------------------------------------!
    !                    Gradient Procedures                        !
    !---------------------------------------------------------------!
                
    ! Gradient Descend
    subroutine GradDesc(self,grad,rate)
        implicit none
        
        class(Tensor),intent(inout) :: self
        class(Tensor),intent(in)    :: grad
        real,intent(in)             :: rate
        
        select case(self%rank)
            case(2)
                self%rank2 = self%rank2 - rate*transpose(grad%rank2)
        end select
    end subroutine
    
    ! Gradient GradLoss
!     function GradLoss(self,expected) result(dX)
!         implicit none
!         
!         class(Tensor),intent(in)    :: self
!         class(Tensor),intent(in)    :: expected
!         type(Tensor)                :: dX
!         
!         select case(self%rank)
!             case(2)
!                 dX = Tensor(size(self%rank2,2),size(self%rank2,1))
!                 
!                 X = 
!         
!         
                
    
end module


program test
    use Tensors
    implicit none

    integer,parameter   :: dp = kind(1.d0)            
    
    type(Tensor) :: A
    type(Tensor) :: B
    
    A = Tensor(2,2)
    write(*,*) A%rank2
    B = 2.0_DP*transpose(A)
        
    write(*,*) B%rank2
end program
