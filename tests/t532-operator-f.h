!-----------------------------------------------------------------------
      subroutine setup_mass(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,&
&           u14,u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,&
&           v16,ierr)
      real*8 ctx
      real*8 u1(1)
      real*8 u2(1)
      real*8 v1(1)
      integer q,ierr

      do i=1,q
        v1(i)=u2(i)*(u1(i+q*0)*u1(i+q*3)-u1(i+q*1)*u1(i+q*2))
      enddo

      ierr=0
      end
!-----------------------------------------------------------------------
      subroutine setup_diff(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,&
&           u14,u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,&
&           v16,ierr)
      real*8 ctx
      real*8 u1(1)
      real*8 u2(1)
      real*8 v1(1)
      real*8 w
      integer q,ierr

      do i=1,q
        w=u2(i)/(u1(i+q*0)*u1(i+q*3)-u1(i+q*1)*u1(i+q*2))
        v1(i+q*0)=w*(u1(i+q*2)*u1(i+q*2)+u1(i+q*3)*u1(i+q*3))
        v1(i+q*1)=w*(u1(i+q*0)*u1(i+q*0)+u1(i+q*1)*u1(i+q*1))
        v1(i+q*2)=-w*(u1(i+q*0)*u1(i+q*2)+u1(i+q*2)*u1(i+q*3))
      enddo

      ierr=0
      end
!-----------------------------------------------------------------------
      subroutine apply(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,&
&           u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,ierr)
      real*8 ctx
      real*8 u1(1)
      real*8 u2(1)
      real*8 u3(1)
      real*8 u4(1)
      real*8 v1(1)
      real*8 v2(1)
      real*8 du0,du1
      integer q,ierr

      do i=1,q
!       mass
        v1(i) = u2(i)*u4(i)
!       diff
        du0=u1(i+q*0)
        du1=u1(i+q*1)
        v2(i+q*0)=u3(i+q*0)*du0+u3(i+q*2)*du1
        v2(i+q*1)=u3(i+q*2)*du0+u3(i+q*1)*du1
      enddo

      ierr=0
      end
!-----------------------------------------------------------------------
      subroutine apply_lin(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,&
&           u14,u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,&
&           v16,ierr)
      real*8 ctx
      real*8 u1(1)
      real*8 u2(1)
      real*8 u3(1)
      real*8 v1(1)
      real*8 v2(1)
      real*8 du0,du1
      integer q,ierr

      do i=1,q
        du0=u1(i+q*0)
        du1=u1(i+q*1)
        v1(i+q*0)=u2(i+q*0)*du0+u2(i+q*3)*du1+u2(i+q*6)*u3(i)
        v2(i+q*0)=u2(i+q*1)*du0+u2(i+q*4)*du1+u2(i+q*7)*u3(i)
        v2(i+q*1)=u2(i+q*2)*du0+u2(i+q*5)*du1+u2(i+q*8)*u3(i)
      enddo

      ierr=0
      end
!-----------------------------------------------------------------------
