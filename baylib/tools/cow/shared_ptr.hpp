//
// Created by elle on 07/08/21.
//

#ifndef BAYLIB_SHARED_PTR_HPP
#define BAYLIB_SHARED_PTR_HPP

/**
 * Adapted version of qshareddata.h from Qt library
 * which already implements the IMPLICIT copy on write
 * in an efficient way
 */

namespace bn{
    namespace cow {
        template <typename T>
        class shared_ptr {
            T *d;

            void detach_helper() {
                T *x = clone();
                ++x->ref;
                if (!--d->ref)
                    delete d;
                d = x;
            }

        protected:
            T* clone(){ return new T(*d); }

        public:
            inline void detach() {
                if (d && d->ref != 1) detach_helper();
            }

            // operators
            inline T& operator * () { detach(); return *d; }
            inline const T& operator*() const { return *d; }

            inline T* operator->() { detach(); return d; }
            inline const T* operator->() const { return d; }

            inline explicit operator T*() { detach(); return d; }
            inline explicit operator const T*() const { return d; }

            inline T* data() { detach(); return d; }
            inline const T* data() const { return d; }

            inline bool operator==(const shared_ptr<T>& other) const { return d == other.d; }
            inline bool operator!=(const shared_ptr<T>& other) const { return d != other.d; }

            // constructors
            shared_ptr() : d(0) {}

            explicit shared_ptr(T* data) : d(data) {
                if(d) ++d->ref;
            }

            ~shared_ptr() {
                if(d && !--d->ref) delete d;
            }

            inline shared_ptr(const shared_ptr<T>& o) : d(o.d) { if (d) ++d->ref; }

            inline shared_ptr<T> & operator=(const shared_ptr<T>& o)
            {
                if (o.d != d)
                {
                    if (o.d)
                        ++o.d->ref;
                    T *old = d;
                    d = o.d;
                    if (old && !--old->ref)
                        delete old;
                }
                return *this;
            }

            inline shared_ptr &operator=(T *o)
            {
                if (o != d)
                {
                    if (o)
                        ++o->ref;
                    T *old = d;
                    d = o;
                    if (old && !--old->ref)
                        delete old;
                }
                return *this;
            }

            inline bool operator!() const { return !d; }

            inline void swap(shared_ptr& other)
            {
                using std::swap;
                swap(d, other.d);
            }

        };
    } // namespace cow
} //namespace bn

#endif //BAYLIB_SHARED_PTR_HPP
